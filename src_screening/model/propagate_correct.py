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
from typing import Union, Callable

# External modules
import xarray as xr
import pandas as pd

# Internal modules
from .propagate import PropagateMEB


logger = logging.getLogger(__name__)


class PropagateCorrect(PropagateMEB):
    def __init__(
            self,
            pipeline: Callable,
            namelist_path: str,
            update_time: str = "10min 8s",
            integration_step: str = "8s",
            output_frequency: int = 1
    ):
        super().__init__(
            namelist_path=namelist_path,
            integration_step=integration_step,
            output_frequency=output_frequency
        )
        self.update_time = pd.to_timedelta(update_time)
        self.pipeline = pipeline

    @staticmethod
    def append_forcing(forcing, model_state: xr.Dataset) -> xr.Dataset:
        wind_speed = forcing.get_windspeed(
            model_state, model_state.time
        )
        state = xr.merge([model_state, wind_speed])
        return state

    def correct(
            self,
            initial_state: xr.Dataset,
            forecast_state: xr.Dataset
    ) -> xr.Dataset:
        ds_output = self.pipeline(initial_state, forecast_state)
        ds_output = ds_output.sinn.enforce_physical_bounds()
        return ds_output

    def _propagation_step(
            self,
            wrapper,
            ds_initial: xr.Dataset,
            lead_time: Union[pd.Timedelta, str]
    ) -> None:
        curr_time = pd.to_datetime(ds_initial.time.values)
        lead_time = pd.to_timedelta(lead_time)
        end_time = curr_time + lead_time
        while curr_time < end_time:
            propagated_state = self._propagate_state(
                wrapper=wrapper, ds_initial=ds_initial,
                lead_time=self.update_time
            )
            initial_state = self.append_forcing(
                wrapper.forcing, ds_initial
            )
            forecast_state = self.append_forcing(
                wrapper.forcing, propagated_state
            )
            ds_initial.close()
            ds_initial = self.correct(initial_state, forecast_state)
            curr_time = pd.to_datetime(ds_initial.time.values)
            initial_state.close()
            forecast_state.close()
            propagated_state.close()
        return None
