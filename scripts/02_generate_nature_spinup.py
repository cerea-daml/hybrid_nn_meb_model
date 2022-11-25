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
parser.add_argument("--namelist_name", type=str, default="config_fine.cfg")
parser.add_argument("--n_workers", type=int, default=48)
parser.add_argument("--cluster_address", type=str, default=None)
parser.add_argument("--end_time", type=str, default="1 days")
parser.add_argument("--integration_step", type=str, default="8s")


def main(args):
    ds_initial = xr.open_zarr(os.path.join(args.data_path, "hr_random_initial"))
    ds_forcing_params = xr.open_dataset(
        os.path.join(args.data_path, "forcing_params.nc")
    )
    ds_initial = ds_initial.compute()
    ds_forcing_params = ds_forcing_params.compute()
    logger.info("Data loaded")

    zarr_store = os.path.abspath(
        os.path.join(args.data_path, "hr_nature_spinup")
    )

    propagation_func = PropagateMEB(
        namelist_path=os.path.join(_model_path, args.namelist_name),
        integration_step=args.integration_step,
        output_frequency=450
    )

    iterator = FixedTimeIterator(
        propagation_func=propagation_func,
    )
    iterator(
        zarr_store=zarr_store,
        ds_initial=ds_initial,
        ds_forcing_params=ds_forcing_params,
        lead_time=args.end_time
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    client = initialize_cluster_client(
        cluster_address=args.cluster_address,
        n_workers=args.n_workers
    )
    main(args)
