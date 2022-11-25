# !/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 19.01.22
#
# Created for subsinn
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}


# System modules
import sys
sys.path.append('../')
import logging
import argparse
import os

# External modules
import xarray as xr

# Internal modules
import utils
from src_screening.utils import initialize_cluster_client


logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--lead_time", type=str, default="10 min 8s")
parser.add_argument("--n_workers", type=int, default=48)
parser.add_argument("--cluster_address", type=str, default=None)


def main(args: argparse.Namespace, data_type: str = "train"):
    data_type_path = os.path.join(args.data_path, data_type)

    # Load data
    ds_forecast = xr.open_zarr(
        os.path.join(data_type_path, "lr_forecast"),
        chunks={"ensemble": -1}
    )
    ds_forecast = ds_forecast.sel(lead_time=["0min 0s", args.lead_time])
    ds_forcing_params = xr.open_dataset(
        os.path.join(data_type_path, "forcing_params.nc")
    )
    ds_forcing = utils.get_forcing_data(
        ds_input=ds_forecast, ds_forcing_params=ds_forcing_params
    )
    logger.info("Loaded datasets")

    # Combine data into input
    ds_input = xr.merge([ds_forecast, ds_forcing])
    ds_input = ds_input.stack(samples=["ensemble", "time"])
    ds_input = ds_input.reset_index("samples", drop=True)
    logger.info("Combined data into input dataset")

    # Create input node data
    ds_input.sinn.node_vars = ["u", "v", "velocity_y"]
    input_nodes = ds_input.sinn.get_nodes_array()
    input_nodes = input_nodes.stack(channels_1=["lead_time", "var_names"])
    input_nodes = input_nodes.reset_index("channels_1", drop=True)
    input_nodes = input_nodes.transpose("samples", "channels_1", "nMesh2_node")
    input_nodes = input_nodes.chunk({
        "samples": 1, "channels_1": -1, "nMesh2_node": -1
    })
    logger.info("Prepared input node data")

    # Create input face data
    input_faces = ds_input.sinn.get_faces_array()
    input_faces = input_faces.stack(channels_2=("lead_time", "var_names"))
    input_faces = input_faces.reset_index("channels_2", drop=True)
    input_faces = input_faces.transpose("samples", "channels_2", "nMesh2_face")
    input_faces = input_faces.chunk({
        "samples": 1, "channels_2": -1, "nMesh2_face": -1
    })
    logger.info("Prepared input face data")

    # Combine data into one dict
    prepared_data = xr.Dataset({
        "nodes": input_nodes,
        "faces": input_faces,
    })
    logger.info("Combined data into one dataset")

    #Normalise data
    mean_path = os.path.join(
        args.data_path, "train", "climatology", "input_normal_mean.nc"
    )
    std_path = os.path.join(
        args.data_path, "train", "climatology", "input_normal_std.nc"
    )
    if data_type == "train":
        # Create mean and stddev
        norm_mean = prepared_data.mean(
            ["samples", "nMesh2_face", "nMesh2_node"]
        )
        norm_std = prepared_data.std(
            ["samples", "nMesh2_face", "nMesh2_node"],
            ddof=1
        )
        logger.info("Created normalisation mean and stddev")
        # Store mean and stddev
        norm_mean.to_netcdf(mean_path, mode="w")
        norm_std.to_netcdf(std_path, mode="w")
        logger.info("Stored normalisation mean and stddev")
    else:
        # Load mean and stddev
        norm_mean = xr.open_dataset(mean_path)
        norm_std = xr.open_dataset(std_path)
        logger.info("Loaded normalisation mean and stddev")
    # Normalise the data
    shifted_data = prepared_data - norm_mean
    normalised_data: xr.Dataset = (shifted_data + 1E-8) / (norm_std + 1E-8)
    logger.info("Normalised data")

    # Store the data to zarr
    normalised_data.to_zarr(
        os.path.join(data_type_path, "dataset", "input_normal"),
        mode="w"
    )
    logger.info("Stored normalised data")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    client = initialize_cluster_client(
        n_workers=args.n_workers,
        memory_limit="8G",
        cluster_address=args.cluster_address
    )
    main(args, "train")
    logger.info("Created train dataset")
    main(args, "eval")
    logger.info("Created eval dataset")
    main(args, "test")
    logger.info("Created test dataset")
    client.close()
