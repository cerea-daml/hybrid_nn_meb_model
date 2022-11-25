#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 21.05.22
#
# Created for Paper SASIP screening
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}


# System modules
import sys
sys.path.append("../../")

import logging
import argparse
import os
from typing import Tuple

# External modules
import xarray as xr
import torch
import dask
from tqdm import tqdm

from hydra.utils import instantiate
from hydra import initialize, compose

# Internal modules
from src_screening.utils import initialize_cluster_client
from src_screening.model import PropagateCorrect, FixedTimeIterator, \
    available_combine_functions, NeuralNetworkPipeline


logger = logging.getLogger(__name__)
torch.set_num_threads(1)


os.environ["OMP_NUM_THREADS"] = "1"
dask.config.set({"distributed.comm.timeouts.tcp": "120s"})
dask.config.set({"distributed.comm.timeouts.connect": "120s"})


_namelist_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "meb_model", "config_coarse.cfg"
)


parser = argparse.ArgumentParser()
parser.add_argument("--processed_path", type=str, required=True)
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--n_workers", type=int, default=64)


def load_model(
        model_checkpoint: str,
) -> Tuple[torch.nn.Module, str, str]:
    model_dir = os.path.dirname(model_checkpoint)
    with initialize(config_path=os.path.join(model_dir, 'hydra')):
        cfg = compose('config.yaml')
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

    input_type = "normal"
    target_type = "normal"
    if "input_type" in cfg["data"].keys():
        input_type = cfg["data"]["input_type"]
        target_type = cfg["data"]["target_type"]
    return model, input_type, target_type


def generate_short_forecast(
        zarr_store: str,
        model_checkpoint: str,
):
    ds_initial = xr.open_zarr("data/raw/test/forecast_data")
    ds_initial = ds_initial.isel(lead_time=0)
    ds_initial = ds_initial.set_index(samples=["ensemble", "time"])
    ds_initial = ds_initial.unstack("samples")
    ds_initial = ds_initial.transpose("ensemble", "time", ...)
    ds_initial = ds_initial.compute()

    ds_forcing_params = xr.open_dataset(
        os.path.join("data/raw/test/forcing_params.nc")
    )
    logger.info("Data loaded")

    network, input_type, target_type = load_model(model_checkpoint)
    logger.info("Network loaded")

    input_climatology = {
        "mean": xr.open_dataset(
            f"data/raw/train/climatology/input_{input_type:s}_mean.nc"
        ),
        "std": xr.open_dataset(
            f"data/raw/train/climatology/input_{input_type:s}_std.nc"
        ),
    }
    target_climatology = {
        "mean": xr.open_dataset(
            f"data/raw/train/climatology/target_{target_type:s}_mean.nc"
        ),
        "std": xr.open_dataset(
            f"data/raw/train/climatology/target_{target_type:s}_std.nc"
        ),
    }
    combine_function = available_combine_functions[input_type]
    pipeline = NeuralNetworkPipeline(
        network=network,
        input_climatology=input_climatology,
        target_climatology=target_climatology,
        combine_function=combine_function
    )
    logger.info("Initialised neural network pipeline")

    propagation_func = PropagateCorrect(
        namelist_path=_namelist_path,
        update_time="10min 8s",
        integration_step="16s",
        output_frequency=1,
        pipeline=pipeline
    )

    try:
        iterator = FixedTimeIterator(propagation_func=propagation_func)
        iterator(
            zarr_store=zarr_store,
            ds_initial=ds_initial,
            ds_forcing_params=ds_forcing_params,
            lead_time="1 hour"
        )
    finally:
        del pipeline


def iterate_through_dirs(
        processed_path: str, path_to_search: str
):
    list_of_files = {}
    for (dirpath, dirnames, filenames) in os.walk(path_to_search):
        for filename in filenames:
            if filename == 'last.ckpt':
                exp_path = dirpath.replace(path_to_search, "")
                list_of_files[exp_path] = os.sep.join([dirpath, filename])
    logger.info(f"Found {len(list_of_files):d} model runs")
    for exp_name, ckpt_path in tqdm(list_of_files.items()):
        logger.info(f"Model checkpoint: {ckpt_path:s}")
        exp_processed_path = os.path.join(processed_path, exp_name)
        zarr_store = os.path.abspath(
            os.path.join(exp_processed_path, "traj_short")
        )
        logger.info(f"Zarr store path: {zarr_store:s}")
        if os.path.isdir(zarr_store):
            logger.info(
                f"{zarr_store:s} already exists, skipping"
            )
        else:
            generate_short_forecast(
                zarr_store=zarr_store,
                model_checkpoint=ckpt_path,
            )


def main(args: argparse.Namespace):
    model_path: str = args.model_path
    if not model_path.endswith("/"):
        model_path = model_path + "/"
    iterate_through_dirs(
        processed_path=args.processed_path,
        path_to_search=model_path,
    )
    logger.info("Stored predictions for all model directories")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    client = initialize_cluster_client(
        n_workers=args.n_workers,
        memory_limit="8GB",
    )
    main(args)
