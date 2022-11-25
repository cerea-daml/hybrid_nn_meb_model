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
from typing import Dict
import argparse
import os

# External modules
import numpy as np
import xarray as xr
from tqdm import tqdm

# Internal modules
import src_screening.model

import utils


logger = logging.getLogger(__name__)

_template_path = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))
    ),
    "data", "interim", "template_hr.nc"
)


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n_ensemble", type=int, default=100)
parser.add_argument("--base_cohesion", type=float, default=5000.)


def generate_forcing_params(rnd: np.random.Generator) -> Dict[str, np.ndarray]:
    amplitude = rnd.uniform(8., 20.)
    base_velocity = np.maximum(rnd.uniform(0, 10.), 20-amplitude)
    advection = rnd.uniform(-0.5, 0.5)
    phase = rnd.uniform(-100000, 100000)
    wave_length = rnd.uniform(50000, 200000)
    forcing_params = {
        'amplitude': np.array((0, amplitude)),
        'base_velocity': np.array((0, base_velocity)),
        'advection_speed': np.array((0, advection)),
        'phase': np.array((0, phase)),
        'wave_length': np.array((wave_length, wave_length)),
    }
    return forcing_params


def generate_params_ds(
        n_ensemble: int, rnd: np.random.Generator
) -> xr.Dataset:
    forcing_params = [generate_forcing_params(rnd) for _ in range(n_ensemble)]
    forcing_params = {
        k: (("ensemble", "coord"), [dic[k] for dic in forcing_params])
        for k in forcing_params[0]
    }
    ds_params = xr.Dataset(forcing_params)
    return ds_params


def generate_random_cohesion(
        n_grid_points: int,
        rnd: np.random.Generator,
        base_cohesion: float = 5000.0,
) -> np.ndarray:
    random_factor = rnd.uniform(1, 2, size=(1, n_grid_points))
    random_cohesion = base_cohesion * random_factor
    return random_cohesion


def construct_ds_initial(
        ds_template: xr.Dataset,
        random_cohesion: np.ndarray
) -> xr.Dataset:
    ds_random = ds_template.sel(time=['1970-01-01'])
    ds_random["cohesion"].data = random_cohesion
    ds_random["u"] = xr.zeros_like(ds_random["u"])
    ds_random["v"] = xr.zeros_like(ds_random["v"])
    ds_random["stress_xx"] = xr.zeros_like(ds_random["stress_xx"])
    ds_random["stress_yy"] = xr.zeros_like(ds_random["stress_yy"])
    ds_random["stress_xy"] = xr.zeros_like(ds_random["stress_xy"])
    ds_random["damage"] = xr.ones_like(ds_random["damage"])
    ds_random["thickness"] = xr.ones_like(ds_random["thickness"])
    ds_random["area"] = xr.ones_like(ds_random["area"])
    return ds_random


def main(args: argparse.Namespace) -> None:
    initial_path = os.path.join(args.data_path, "hr_random_initial")
    forcing_path = os.path.join(args.data_path, "forcing_params.nc")

    rnd = np.random.default_rng(args.seed)
    logger.info("Seeded the initial random number generator")

    forcing_params = generate_params_ds(args.n_ensemble, rnd)
    forcing_params.to_netcdf(forcing_path)
    logger.info("Generated forcing parameters")

    ds_template = xr.open_dataset(_template_path).isel(time=[0])
    ds_template = ds_template.drop_vars(
        [var for var in ds_template.data_vars.keys() if var.startswith("f_")]
    )
    logger.info("Loaded the template dataset")

    list_of_initial = []
    for _ in tqdm(range(args.n_ensemble)):
        random_cohesion = generate_random_cohesion(
            ds_template["nMesh2_face"].size, rnd, args.base_cohesion
        )
        ds_member_initial = construct_ds_initial(
            ds_template, random_cohesion
        )
        list_of_initial.append(ds_member_initial)
    logger.info("Looped through all members, start with the computing")
    ds_initial = xr.concat(list_of_initial, dim="ensemble")
    ds_initial = ds_initial.assign_coords(
        ensemble=np.arange(len(ds_initial.ensemble))
    )
    ds_initial = ds_initial.chunk(
        {"ensemble": 1, "time": -1}
    )
    ds_initial.to_zarr(initial_path, mode="w")
    logger.info("Finished writing the datasets")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    main(args)
