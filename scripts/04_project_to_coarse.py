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
from src_screening.model.fem_interpolation import FEMInterpolation
from src_screening.model.post_processing import post_process_grid
from src_screening.utils import initialize_cluster_client
import utils


_template_path = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))
    ),
    "data", "interim", "template_lr.nc"
)


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--n_workers", type=int, default=1)


logger = logging.getLogger(__name__)


def project_to_lr(
        ds_hr_run: xr.Dataset,
        ds_lr_template: xr.Dataset
) -> xr.Dataset:
    ds_hr_run = post_process_grid(ds_hr_run)
    projector = FEMInterpolation(
        ds_origin=ds_hr_run,
    )
    ds_projected = projector.interpolate(ds_lr_template)
    ds_projected = ds_projected.chunk({"time": 1000, "ensemble": 1})
    ds_projected = ds_projected.sinn.convert_mesh_to_coords()
    ds_projected = ds_projected.sinn.get_saveable_ds()
    return ds_projected


def main(args):
    proj_path = os.path.join(args.data_path, "lr_nature_forecast")

    ds_template = utils.load_nc_dataset(_template_path).isel(time=[0])
    ds_template = ds_template.drop_vars(
        [var for var in ds_template.data_vars.keys() if var.startswith("f_")]
    )
    ds_template = ds_template.chunk({"time": 1})
    logger.info("Loaded the template dataset")

    ds_nature_target = xr.open_zarr(
        os.path.join(args.data_path, "hr_nature_forecast")
    )
    logger.info("Loaded the nature target dataset")

    ds_proj_target = project_to_lr(ds_nature_target, ds_template)
    logger.info("Project the nature target dataset")

    ds_proj_target.to_zarr(proj_path, mode="w", compute=True)
    logger.info("Stored the target data")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    client = initialize_cluster_client(n_workers=args.n_workers)
    main(args)
