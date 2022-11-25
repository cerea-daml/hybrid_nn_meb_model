#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 19.01.22
#
# Created for Paper NN Screening sea-ice
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import Dict, Union

# External modules
import xarray as xr
import pandas as pd
import numpy as np
import dask.array as da
import dask

from distributed import Client, LocalCluster

# Internal modules
from src_screening.model.post_processing import post_process_grid
from src_screening.model.wave_forcing import WaveForcing


logger = logging.getLogger(__name__)


def load_nc_dataset(path: str) -> xr.Dataset:
    """
    Loads a netcdf from specified path and post-process its grid.

    Parameters
    ----------
    path : str
        The path to the netcdf file.

    Returns
    -------
    xr.Dataset
        The loaded netcdf dataset with the post-processed grid.
    """
    ds_loaded = xr.open_dataset(path, chunks={'time': 100})
    ds_loaded = post_process_grid(ds_loaded)
    return ds_loaded


def write_forcing_template(
        forcing_path: str,
        node_coordinates: pd.MultiIndex,
        time_index: pd.DatetimeIndex,
        n_ensemble: int,
) -> None:
    forcing_data = xr.DataArray(
        data=da.random.random(
            (n_ensemble, time_index.size, node_coordinates.size),
            chunks=(1, 100, -1)
        ),
        dims=["ensemble", "time", "nMesh2_node"],
    )
    attr_data = xr.DataArray(
        data=da.random.random(
            (n_ensemble,),
            chunks=(1, )
        ),
        dims=["ensemble"],

    )
    ds_forcing = xr.Dataset(
        {
            'forcing_y': forcing_data.copy(deep=False),
            'forcing_x': forcing_data.copy(deep=False),
            'amplitude': attr_data.copy(deep=False),
            'base_velocity': attr_data.copy(deep=False),
            'advection': attr_data.copy(deep=False),
            'phase': attr_data.copy(deep=False),
            'wave_length': attr_data.copy(deep=False),
        },
        coords={
            'time': time_index,
            'nMesh2_node': node_coordinates,
            'ensemble': np.arange(n_ensemble),
        },
    )
    ds_forcing = ds_forcing.reset_index("nMesh2_node")
    ds_forcing.to_zarr(forcing_path, mode="w", compute=False)
    logger.info("Created forcing output template")


def generate_forcing_in_ydir(
        coordinates: np.ndarray,
        time_index: pd.DatetimeIndex,
        base_velocity: float = 10.,
        amplitude: float = 10.,
        advection: float = 0.1,
        phase: float = 50000,
        wave_length: float = 100000.,
        air_drag_coeff: float = 1.5E-3,
        air_density: float = 1.3,
) -> np.ndarray:
    lead_time = time_index - pd.Timestamp('1970-01-01 00:00:00')
    sine_factor = lead_time.total_seconds().values[:, None] * advection
    sine_factor = sine_factor + coordinates[None, :] + phase
    sine_wave = amplitude * np.sin(2 * np.pi * sine_factor / wave_length)
    time_scaling = (lead_time/pd.Timedelta(days=1)).values[:, None]
    time_scaling = np.minimum(1., time_scaling)
    velocity = (base_velocity + sine_wave) * time_scaling
    forcing = velocity * np.abs(velocity) * air_drag_coeff * air_density
    return forcing


def construct_ds_forcing(
        node_coordinates: pd.MultiIndex,
        time_index: pd.DatetimeIndex,
        forcing_y: np.ndarray,
        forcing_params: Dict[str, float],
) -> xr.Dataset:
    ds_forcing = xr.Dataset(
        {
            'forcing_y': (("time", "nMesh2_node"), forcing_y),
            'forcing_x': (("time", "nMesh2_node"), np.zeros_like(forcing_y),),
            **forcing_params
        },
        coords={
            'time': time_index,
            'nMesh2_node': node_coordinates
        },
    )
    return ds_forcing


@dask.delayed
def construct_forcing(
        node_coordinates: pd.MultiIndex,
        time_index: pd.DatetimeIndex,
        forcing_params: Dict[str, float],
) -> xr.Dataset:
    forcing_y = generate_forcing_in_ydir(
        node_coordinates.get_level_values("Mesh2_node_y").values, time_index,
        **forcing_params
    )
    ds_forcing = construct_ds_forcing(
        node_coordinates, time_index, forcing_y, forcing_params
    )
    ds_forcing = ds_forcing.chunk({"time": 100})
    ds_forcing = ds_forcing.expand_dims("ensemble", axis=0)
    return ds_forcing


@dask.delayed
def store_forcing(
        idx_mem: int,
        ds_forcing: xr.Dataset,
        store_path: str
) -> None:
    ds_saveable: xr.Dataset = ds_forcing.reset_index("nMesh2_node")
    ds_saveable = ds_saveable.drop_vars(
        ["Mesh2_node_x", "Mesh2_node_y", "time"]
    )
    ds_saveable.to_zarr(
        store_path, mode="r+",
        region={"ensemble": slice(idx_mem, idx_mem+1)},
    )


def construct_time_index(
        end_time: str,
        time_spacing: str,
        start_time: Union[str, None] = None
) -> pd.DatetimeIndex:
    base_time = pd.to_datetime('1970-01-01 00:00:00')
    if start_time is None:
        start_time = '00:00:00'
    start_time = base_time + pd.to_timedelta(start_time)
    end_time = base_time + pd.to_timedelta(end_time)
    time_spacing = pd.to_timedelta(time_spacing)
    return pd.date_range(start_time, end_time, freq=time_spacing)


def get_forcing_data(
        ds_input: xr.Dataset,
        ds_forcing_params: xr.Dataset,
) -> xr.Dataset:
    forcing = WaveForcing(
        **ds_forcing_params.data_vars
    )
    time_array = ds_input["time"]
    if "lead_time" in ds_input.dims:
        time_array = time_array + ds_input["lead_time"]
    ds_forcing = forcing.get_windspeed(
        ds_input,
        time_array
    )
    return ds_forcing
