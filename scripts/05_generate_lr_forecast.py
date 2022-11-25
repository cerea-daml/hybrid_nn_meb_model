#!/bin/env python
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
from src_screening.model import FixedTimeIterator, PropagateMEB
from src_screening.utils import initialize_cluster_client
import utils


logger = logging.getLogger(__name__)


os.environ["OMP_NUM_THREADS"] = "1"


_model_path = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))
    ),
    "meb_model",
)


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--namelist_name", type=str, default="config_coarse.cfg")
parser.add_argument("--n_workers", type=int, default=1)
parser.add_argument("--start_time", type=str, default="1 day")
parser.add_argument("--final_time", type=str, default="2 days 23 hour")
parser.add_argument("--lead_time", type=str, default="1 hour")
parser.add_argument("--time_spacing", type=str, default="1 hour")
parser.add_argument("--integration_step", type=str, default="16s")


def main(args):
    projected_ds = xr.open_zarr(
        os.path.join(args.data_path, "lr_nature_forecast")
    )
    projected_ds = projected_ds.stack(combined_time=["time", "lead_time"])
    projected_ds = projected_ds.assign_coords(
        combined_time=projected_ds["time"] + projected_ds["lead_time"])
    projected_ds = projected_ds.rename({"combined_time": "time"})
    ds_forcing_params = xr.open_dataset(
        os.path.join(args.data_path, "forcing_params.nc")
    )
    logger.info("Data loaded")

    initial_time_index = utils.construct_time_index(
        start_time=args.start_time,
        end_time=args.final_time,
        time_spacing=args.time_spacing
    )
    ds_initial = projected_ds.sel(
        time=initial_time_index, method="nearest"
    )
    ds_initial = ds_initial.compute()
    logger.info("Initial data generated")

    zarr_store = os.path.abspath(
        os.path.join(args.data_path, "lr_forecast")
    )
    propagation_func = PropagateMEB(
        namelist_path=os.path.join(_model_path, args.namelist_name),
        integration_step=args.integration_step,
    )
    iterator = FixedTimeIterator(
        propagation_func=propagation_func,
    )
    iterator(
        zarr_store=zarr_store,
        ds_initial=ds_initial,
        ds_forcing_params=ds_forcing_params,
        lead_time=args.lead_time
    )
    logger.info("Data propagated")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    _ = initialize_cluster_client(n_workers=args.n_workers)
    main(args)
