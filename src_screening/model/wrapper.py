#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 28.03.22
#
# Created for subsinn
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}


# System modules
import logging
import os.path
import tempfile
from typing import Union, List, Iterable

# External modules
import numpy as np
import xarray as xr
import pandas as pd
import coutcatcher

import pyMEB

# Internal modules
from .wave_forcing import WaveForcing
from .post_processing import post_process_grid


logger = logging.getLogger(__name__)


class MEBWrapper(object):
    """
    This is a xarray wrapper around the PyMEB interface to the
    maxwell-elasto-brittle sea-ice dynamics only model by Dansereau et al.
    2016.
    This wrapper works directly on xarray Datasets and represents the current
    `state` within the model.
    This state can be then `propagated` by the model to a given lead time.
    Along its way, the model might generate a `model output` as netCDF file.
    This netCDF-file is a temporary file and stored as long as this wrapper
    is instantiated.
    """
    def __init__(
            self,
            namelist_path: str,
            integration_delta: Union[pd.Timedelta, str] = "16s",
            output_frequency: int = 1,
            forcing: Union[WaveForcing, None] = None,
            call_args: Iterable[str] = None,
            keep_output: bool = False
    ):
        self._time: pd.Timestamp = pd.Timestamp("1970-01-01 00:00:00")
        self._forcing = None
        self._state: xr.Dataset = None
        self._dg = None
        self._post = None
        self._output = None
        self._integration_delta = None
        self._output_file = tempfile.NamedTemporaryFile(
            suffix=".nc", delete=~keep_output
        )

        self.forcing = forcing
        self.namelist_path = namelist_path
        self.integration_delta = integration_delta
        self.output_frequency = output_frequency
        self.call_args = call_args
        self.state_list = [
            "u", "v", "stress_xx", "stress_xy", "stress_yy", "damage",
            "cohesion", "area", "thickness"
        ]
        self.set_model()

    def close(self):
        self._output_file.close()
        if os.path.isfile(self.output_path):
            os.remove(self.output_path)
        del self._dg
        del self._post
        del self._output
        self._dg = None
        self._post = None
        self._output = None

    def __del__(self):
        self.close()

    @property
    def output_path(self) -> str:
        return self._output_file.name

    @property
    def output(self) -> xr.Dataset:
        loaded_state = xr.open_dataset(self.output_path).compute()
        loaded_state = post_process_grid(loaded_state)
        loaded_state = self._construct_time_multiindex(loaded_state)
        loaded_state = loaded_state.sinn.get_saveable_ds()
        return loaded_state

    @property
    def state(self) -> xr.Dataset:
        if self._state is None:
            raise ValueError("State has to be set.")
        return self._state

    @state.setter
    def state(self, state: xr.Dataset):
        try:
            self._state.close()
        except AttributeError:
            pass
        state_vars = self._get_var_intersection(state.data_vars.keys())
        for var_name in state_vars:
            state_vals = state[var_name].values.squeeze().copy()
            assert state_vals.ndim == 1, "State must be 1D."
            pyMEB.assignValue(
                self._dg, var_name, state_vals
            )
        self._state = state
        self._time = pd.to_datetime(state.time.values)
        pyMEB.adjustModelConsistency(self._dg, self.time_step)

    @property
    def forcing(self):
        return self._forcing

    @forcing.setter
    def forcing(self, forcing: Union[WaveForcing, None]):
        if forcing is not None and not isinstance(forcing, WaveForcing):
            raise TypeError("Forcing must be of type Forcing or None.")
        self._forcing = forcing

    @property
    def integration_delta(self) -> pd.Timedelta:
        return self._integration_delta

    @integration_delta.setter
    def integration_delta(self, new_time: Union[pd.Timedelta, str]):
        self._integration_delta = pd.Timedelta(new_time)

    @property
    def time(self) -> Union[pd.Timestamp, None]:
        return self._time

    @property
    def time_step(self) -> int:
        run_time = self._time - pd.Timestamp("1970-01-01")
        run_time = run_time.total_seconds()
        time_step = int(run_time / self.integration_delta.total_seconds())
        return time_step

    @staticmethod
    def _construct_time_multiindex(ds: xr.Dataset) -> xr.Dataset:
        ds_time = ds.time.values
        time_multiindex = pd.MultiIndex.from_product(
            [[ds_time[0]], ds_time - ds_time[0]],
            names=["time", "lead_time"]
        )
        ds = ds.rename({"time": "combined_time"})
        ds = ds.assign_coords(combined_time=time_multiindex)
        ds = ds.unstack("combined_time")
        ds = ds.transpose("time", "lead_time", ...)
        return ds

    def _construct_model_args(self, namelist_name: str) -> List[str]:
        output_dir, output_file = os.path.split(self.output_path)
        model_dt = int(self.integration_delta.total_seconds())
        model_args = [
            "MEB_model",
            f"--config={namelist_name}",
            f"--resolution.dt={str(model_dt)}",
            "--output.format=netCDF",
            f"--output.dirname={output_dir}",
            f"--output.filename={output_file}",
            f"--output.frequency={str(self.output_frequency)}",
            f"--input.dirname={output_dir}"
        ]
        if self.forcing is not None:
            model_args += [
                "--forcing.forcing-type=External"
            ]
        if self.call_args is not None:
            model_args += list(self.call_args)
        return model_args

    def _get_var_intersection(self, var_list: Iterable[str]) -> List[str]:
        state_vars = [
            k for k in var_list if k in self.state_list
        ]
        return state_vars

    def set_model(self):
        self.close()
        model_folder, namelist_name = os.path.split(self.namelist_path)
        curr_path = os.getcwd()
        os.chdir(model_folder)
        model_args = self._construct_model_args(namelist_name)
        with coutcatcher.capture():
            self._dg, self._post, self._output, _, _ = pyMEB.initModel(
                model_args
            )
        os.chdir(curr_path)

    def _get_velocity_coords(self) -> xr.Dataset:
        x, y = pyMEB.getVarCoordinate(self._dg, "u")
        ds_coords = xr.Dataset(coords={
            "Mesh2_node_x": ("nMesh2_node", x),
            "Mesh2_node_y": ("nMesh2_node", y),
        })
        return ds_coords

    def _update_forcing(self) -> None:
        ds_coords = self._get_velocity_coords()
        ds_forcing = self.forcing.get_windspeed(ds_coords, self.time)
        pyMEB.assignValue(
            self._dg, "forcing_x", ds_forcing.velocity_x.values,
        )
        pyMEB.assignValue(
            self._dg, "forcing_y", ds_forcing.velocity_y.values
        )

    def propagate_single_step(self) -> None:
        self._time += self.integration_delta
        if self.forcing is not None:
            self._update_forcing()
        pyMEB.stepModel(self._dg, self._post, self._output, 0)

    def update_output(self) -> None:
        self._output.update(self._dg, self._post)

    def update_model_state(self) -> None:
        new_dataset = self._state.copy()
        new_dataset["time"] = self._time
        state_vars = self._get_var_intersection(new_dataset.data_vars.keys())
        for var_name in state_vars:
            new_array = np.zeros_like(self._state[var_name].values.squeeze())
            pyMEB.getValue(self._dg, var_name, new_array)
            new_array = np.broadcast_to(
                new_array, shape=new_dataset[var_name].shape
            )
            new_dataset[var_name] = new_dataset[var_name].copy(data=new_array)
        self._state = new_dataset

    def propagate(
            self,
            end_time: pd.Timestamp,
    ) -> None:
        while self.time < end_time:
            self.propagate_single_step()
            self.update_output()
        self.update_model_state()

    def run(self, lead_time: Union[pd.Timedelta, str] = "16s"):
        lead_time = pd.Timedelta(lead_time)
        self.update_output()
        self.propagate(self.time + lead_time)
