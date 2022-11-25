#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 25.05.22
#
# Created for Paper SASIP screening
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}


# System modules
import logging

# External modules
import xarray as xr

# Internal modules
from .accessor import SinnAccessor


logger = logging.getLogger(__name__)


def combine_normal(
        ds_initial: xr.Dataset,
        ds_forecast: xr.Dataset
) -> xr.Dataset:
    ds_input = xr.concat([ds_initial, ds_forecast], dim="time")
    ds_input.sinn.node_vars = ["u", "v", "velocity_y"]
    return ds_input


def combine_initial(
        ds_initial: xr.Dataset,
        ds_forecast: xr.Dataset
) -> xr.Dataset:
    ds_input = ds_initial.copy()
    ds_input.sinn.node_vars = ["u", "v", "velocity_y"]
    return ds_input


def combine_forecast(
        ds_initial: xr.Dataset,
        ds_forecast: xr.Dataset
) -> xr.Dataset:
    ds_input = ds_forecast.copy()
    ds_input.sinn.node_vars = ["u", "v", "velocity_y"]
    return ds_input


def combine_woforcing(
        ds_initial: xr.Dataset,
        ds_forecast: xr.Dataset
) -> xr.Dataset:
    ds_input = xr.concat([ds_initial, ds_forecast], dim="time")
    ds_input = ds_input.drop_vars("velocity_y")
    ds_input.sinn.node_vars = ["u", "v"]
    return ds_input


def combine_difference(
        ds_initial: xr.Dataset,
        ds_forecast: xr.Dataset
) -> xr.Dataset:
    ds_input = xr.concat([ds_initial, ds_forecast], dim="time")
    ds_input[{"time": 1}] = ds_input.isel(time=1)-ds_input.isel(time=0)
    ds_input.sinn.node_vars = ["u", "v", "velocity_y"]
    return ds_input


def combine_fcst_difference(
        ds_initial: xr.Dataset,
        ds_forecast: xr.Dataset
) -> xr.Dataset:
    ds_input = xr.concat([ds_initial, ds_forecast], dim="time")
    ds_input[{"time": 0}] = ds_input.isel(time=1)-ds_input.isel(time=0)
    ds_input.sinn.node_vars = ["u", "v", "velocity_y"]
    return ds_input
