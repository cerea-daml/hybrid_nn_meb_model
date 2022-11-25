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
    ds_forecast = ds_forecast.sel(lead_time=args.lead_time)
    ds_nature = xr.open_zarr(
        os.path.join(data_type_path, "lr_nature_forecast"),
        chunks={"ensemble": -1}
    ).sel(lead_time=args.lead_time)
    logger.info("Loaded datasets")

    # Reindex time for ds_nature
    stacked_forecast = ds_forecast.stack(combined_time=["time", "lead_time"])
    ds_nature = ds_nature.reindex(
        combined_time=stacked_forecast["time"] + stacked_forecast["lead_time"]
    )
    ds_nature = ds_nature.assign_coords(
        combined_time=stacked_forecast["combined_time"]
    )
    ds_nature = ds_nature.unstack("combined_time")
    logger.info("Reindexed nature time")

    # Combine data for error
    ds_error = ds_nature-ds_forecast
    ds_error = ds_error.stack(samples=["ensemble", "time"])
    ds_error = ds_error.reset_index("samples")
    logger.info("Combined data into error dataset")

    # Create error node data
    error_nodes = ds_error.sinn.get_nodes_array()
    error_nodes = error_nodes.rename({"var_names": "var_names_1"})
    error_nodes = error_nodes.reset_index("var_names_1", drop=True)
    error_nodes = error_nodes.transpose("samples", "var_names_1", "nMesh2_node")
    error_nodes = error_nodes.chunk({
        "samples": 1, "var_names_1": -1, "nMesh2_node": -1
    })
    logger.info("Prepared error node data")

    # Create error face data
    error_faces = ds_error.sinn.get_faces_array()
    error_faces = error_faces.rename({"var_names": "var_names_2"})
    error_faces = error_faces.reset_index("var_names_2", drop=True)
    error_faces = error_faces.transpose("samples", "var_names_2", "nMesh2_face")
    error_faces = error_faces.chunk({
        "samples": 1, "var_names_2": -1, "nMesh2_face": -1
    })
    logger.info("Prepared error face data")

    # Combine data into one dict
    prepared_data = xr.Dataset({
        "nodes": error_nodes,
        "faces": error_faces
    })
    logger.info("Combined target data into one dataset")

    #Normalise data
    mean_path = os.path.join(
        args.data_path, "train", "climatology", "target_normal_mean.nc"
    )
    std_path = os.path.join(
        args.data_path, "train", "climatology", "target_normal_std.nc"
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
        os.path.join(data_type_path, "dataset", "target_normal"),
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
