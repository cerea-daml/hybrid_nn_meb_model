#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 14.03.22
#
# Created for Paper SASIP screening
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}


# System modules
import logging
import argparse
from typing import Tuple, Dict
import os
from copy import deepcopy

# External modules
import torch.nn
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import DictConfig

from tqdm.autonotebook import tqdm
import xarray as xr
import numpy as np
from distributed import Client, LocalCluster
import dask

# Internal modules
import src_screening.model

from src_screening.data_module import OfflineDataModule


logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--processed_path", type=str, required=True)
parser.add_argument("--n_workers", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=128)


def load_datamodule(input_type: str):
    data_module: OfflineDataModule = OfflineDataModule(
        data_path="data/raw/",
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
        input_type=input_type,
        target_type="normal"
    )
    data_module.setup()
    return data_module


def load_model(
        model_checkpoint: str,
        cfg: DictConfig,
) -> torch.nn.Module:
    # To support old models
    if "model" in cfg.keys():
        cfg["model"]["_target_"] = cfg["model"]["_target_"].replace(
            ".model.", ".network."
        )
        cfg["model"]["backbone"]["_target_"] = cfg["model"]["backbone"]["_target_"].replace(
            ".model.", ".network."
        )
        model = instantiate(
            cfg.model,
            optimizer_config=cfg.optimizer,
            _recursive_=False
        )
    else:
        model = instantiate(
            cfg.network,
            optimizer_config=cfg.optimizer,
            _recursive_=False
        )
    state_dict = torch.load(model_checkpoint, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict["state_dict"], strict=False)
    model = model.eval().cpu()
    return model


@dask.delayed(nout=2)
def predict(
        batch: Dict[str, torch.Tensor],
        model: torch.nn.Module
) -> Tuple[np.ndarray, np.ndarray]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    with torch.no_grad():
        predicted_nodes, predicted_faces = model(
            batch["input_nodes"].to(device),
            batch["input_faces"].to(device),
        )
    predicted_nodes = predicted_nodes.cpu().numpy()
    predicted_faces = predicted_faces.cpu().numpy()
    return predicted_nodes, predicted_faces


def get_predictions(
        data_module: OfflineDataModule,
        model: torch.nn.Module,
) -> Tuple[np.ndarray, np.ndarray]:
    test_loader = data_module.test_dataloader()
    delayed_model = dask.delayed(model)
    all_nodes = []
    all_faces = []
    for batch in test_loader:
        curr_nodes, curr_faces = predict(batch, delayed_model)
        all_nodes.append(curr_nodes)
        all_faces.append(curr_faces)
    delayed_concat = dask.delayed(np.concatenate)
    concatenated_nodes = delayed_concat(all_nodes, axis=0)
    concatenated_faces = delayed_concat(all_faces, axis=0)
    return concatenated_nodes, concatenated_faces


@dask.delayed
def store_predictions(
        processed_path: str,
        prediction_nodes: np.ndarray,
        prediction_faces: np.ndarray
) -> str:
    # Load the necessary data
    ds_forecast = xr.open_zarr("data/raw/test/lr_forecast/")
    ds_forecast = ds_forecast.sel(lead_time="10min 8s")
    ds_stacked = ds_forecast.stack(samples=["ensemble", "time"])
    normalisation = {
        "mean": xr.open_dataset("data/raw/train/offline_mean.nc"),
        "std": xr.open_dataset("data/raw/train/offline_std.nc"),
    }
    logger.info("Loaded the necessary data for output")

    # Prepare array for node predictions
    ds_nodes = ds_stacked.sinn.get_nodes_array()
    ds_nodes = ds_nodes.rename({"var_names": "var_names_1"})
    ds_nodes = ds_nodes.transpose(
        "samples", "var_names_1", "nMesh2_node"
    )

    # Copy to correction, denormalise, and get dataset
    ds_nodes_pred = ds_nodes.copy(data=prediction_nodes)
    ds_nodes_pred = ds_nodes_pred * normalisation["std"]["error_nodes"]
    ds_nodes_pred = ds_nodes_pred + normalisation["mean"]["error_nodes"]
    ds_nodes_pred = ds_nodes_pred + ds_nodes
    ds_nodes_pred = ds_nodes_pred.to_dataset("var_names_1")
    logger.info("Created node predictions")

    # Prepare array for face predictions
    ds_faces = ds_stacked.sinn.get_faces_array()
    ds_faces = ds_faces.rename({"var_names": "var_names_2"})
    ds_faces = ds_faces.transpose(
        "samples", "var_names_2", "nMesh2_face"
    )

    # Copy to correction, denormalise, and get dataset
    ds_faces_pred = ds_faces.copy(data=prediction_faces)
    ds_faces_pred = ds_faces_pred * normalisation["std"]["error_faces"]
    ds_faces_pred = ds_faces_pred + normalisation["mean"]["error_faces"]
    ds_faces_pred = ds_faces_pred + ds_faces
    ds_faces_pred = ds_faces_pred.to_dataset("var_names_2")
    logger.info("Created face predictions")

    # Create prediction dataset
    ds_prediction = xr.merge([ds_nodes_pred, ds_faces_pred])
    ds_prediction = ds_prediction.unstack("samples")
    ds_prediction = ds_prediction.transpose("ensemble", "time", ...)
    logger.info("Created prediction dataset")

    # Store prediction dataset to zarr
    store_path = os.path.join(processed_path, "prediction_offline")
    ds_prediction.to_zarr(store_path, mode="w")
    return store_path


def model_pipeline(model_checkpoint: str, processed_path: str):
    model_dir = os.path.dirname(model_checkpoint)
    with initialize(config_path=os.path.join(model_dir, 'hydra')):
        cfg = compose('config.yaml')
    logger.info(f"Loaded config for {model_dir:s}")

    data_module = load_datamodule(cfg.data.input_type)
    logger.info("Loaded data module")

    model = load_model(
        model_checkpoint,
        cfg=cfg,
    )
    logger.info("Loaded model")

    prediction_nodes, prediction_faces = get_predictions(
        data_module, model
    )
    logger.info("Got prediction data")

    os.makedirs(processed_path, exist_ok=True)
    store_task = store_predictions(
        processed_path,
        prediction_nodes=prediction_nodes,
        prediction_faces=prediction_faces
    )
    store_task.compute()
    logger.info(f"Stored prediction at {processed_path:s}")
    del model
    logger.info("Removed the model")


def iterate_through_dirs(processed_path: str, path_to_search: str):
    list_of_files = {}
    for (dirpath, dirnames, filenames) in os.walk(path_to_search):
        for filename in filenames:
            if filename == 'last.ckpt':
                exp_path = dirpath.replace(path_to_search, "")
                list_of_files[exp_path] = os.sep.join([dirpath, filename])
    logger.info(f"Found {len(list_of_files):d} model runs")
    for exp_name, ckpt_path in tqdm(list_of_files.items()):
        exp_processed_path = os.path.join(processed_path, exp_name)
        model_pipeline(
            model_checkpoint=ckpt_path,
            processed_path=exp_processed_path
        )


def main(args: argparse.Namespace):
    cluster = LocalCluster(
        n_workers=args.n_workers, threads_per_worker=1, local_directory="/tmp"
    )
    client = Client(cluster)
    logger.info("Dashboard: %s", client.dashboard_link)

    logger.info("Loaded data")

    model_path: str = args.model_path
    if not model_path.endswith("/"):
        model_path = model_path + "/"

    iterate_through_dirs(
        processed_path=args.processed_path,
        path_to_search=model_path
    )
    logger.info("Stored predictions for all model directories")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    main(args)
