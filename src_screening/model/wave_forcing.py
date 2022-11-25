#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 01.04.22
#
# Created for subsinn
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import Union

# External modules
import xarray as xr
import pandas as pd
import numpy as np

# Internal modules


logger = logging.getLogger(__name__)


class WaveForcing(object):
    def __init__(
            self,
            amplitude: Union[float, xr.DataArray] = 1.0,
            base_velocity: Union[float, xr.DataArray] = 0.0,
            advection_speed: Union[float, xr.DataArray] = 0.0,
            phase: Union[float, xr.DataArray] = 0.0,
            wave_length: Union[float, xr.DataArray] = 1.0,
            initial_delay: Union[str, pd.Timedelta] = "1d",
    ):
        super().__init__()
        self.amplitude = amplitude
        self.base_velocity = base_velocity
        self.advection_speed = advection_speed
        self.phase = phase
        self.wave_length = wave_length
        self.initial_delay = pd.to_timedelta(initial_delay).to_timedelta64()

    @staticmethod
    def extract_node_array(coords_nodes: xr.Dataset) -> xr.DataArray:
        node_array = xr.concat(
            (
                coords_nodes["Mesh2_node_x"],
                coords_nodes["Mesh2_node_y"]
            ),
            dim="coord"
        )
        node_array["coord"] = ["x", "y"]
        return node_array

    def estimate_vel_array(
            self,
            node_array: xr.DataArray,
            time: [np.datetime64, pd.Timestamp, xr.DataArray],
    ) -> xr.DataArray:
        if isinstance(time, pd.Timestamp):
            time = time.to_datetime64()
        run_time = time - np.datetime64("1970-01-01 00:00:00")

        run_time_delay = run_time / self.initial_delay
        time_scaling = np.minimum(1., run_time_delay)

        run_time_secs = run_time / np.timedelta64(1, "s")

        time_term = run_time_secs * self.advection_speed
        factor = (time_term + node_array + self.phase) / self.wave_length
        velocity = self.amplitude * np.sin(factor * 2 * np.pi)
        velocity = velocity + self.base_velocity
        velocity = velocity * time_scaling
        return velocity

    def get_windspeed(
            self,
            coords_nodes: xr.Dataset,
            time: [np.datetime64, pd.Timestamp, xr.DataArray],
    ) -> xr.Dataset:
        """
        This abstract method estimates the wind speed for given time and
        coordinates.

        Parameters
        ----------
        coords_nodes: xr.Dataset
            This dataset has to contain the coordinates of the nodes.
        time: pd.DatetimeIndex

        Returns
        -------
        forcing: xr.Dataset
            The estimated forcing based on given coordinates and time.
            The arrays in `x`-direction and `y`-direction are called:
            `velocity_x` and `velocity_y`.
        """
        node_array = self.extract_node_array(coords_nodes)
        vel_array = self.estimate_vel_array(node_array, time)
        ds_windspeed = vel_array.to_dataset(dim="coord")
        ds_windspeed = ds_windspeed.rename(
            {"x": "velocity_x", "y": "velocity_y"}
        )
        ds_windspeed = ds_windspeed.assign(
            velocity_x=ds_windspeed["velocity_x"].transpose(
                ..., "nMesh2_node"
            ),
            velocity_y=ds_windspeed["velocity_y"].transpose(
                ..., "nMesh2_node"
            ),
        )
        ds_windspeed = ds_windspeed.astype(np.float64, order="C")
        return ds_windspeed