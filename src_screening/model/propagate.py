#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 22.08.22
#
# Created for Paper SASIP screening
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import List, Callable, Union
import os

# External modules
import numpy as np
import xarray as xr
import pandas as pd
import dask

from zarr import TempStore

# Internal modules
from .wave_forcing import WaveForcing
from .wrapper import MEBWrapper


logger = logging.getLogger(__name__)


class PropagateMEB(object):
    def __init__(
            self,
            namelist_path: str,
            integration_step: str = "8s",
            output_frequency: int = 1,
            **kwargs
    ):
        self.namelist_path = namelist_path
        self.integration_step = integration_step
        self.output_frequency = output_frequency

    @staticmethod
    def _propagate_state(
            wrapper: MEBWrapper,
            ds_initial: xr.Dataset,
            lead_time: Union[pd.Timedelta, str]
    ) -> xr.Dataset:
        wrapper.state = ds_initial
        wrapper.run(lead_time=lead_time)
        return wrapper.state

    def _propagation_step(
            self,
            wrapper,
            ds_initial: xr.Dataset,
            lead_time: Union[pd.Timedelta, str]
    ) -> None:
        _ = self._propagate_state(
            wrapper=wrapper,
            ds_initial=ds_initial,
            lead_time=lead_time
        )

    def _propagate_to_tempstore(
            self,
            ds_initial: xr.Dataset,
            ds_forcing_params: xr.Dataset,
            lead_time: Union[pd.Timedelta, str]
    ) -> TempStore:
        wrapper = MEBWrapper(
            namelist_path=self.namelist_path,
            forcing=WaveForcing(
                **ds_forcing_params.data_vars
            ),
            integration_delta=self.integration_step,
            output_frequency=self.output_frequency,
            keep_output=False
        )
        try:
            _ = self._propagation_step(
                wrapper=wrapper, ds_initial=ds_initial, lead_time=lead_time
            )
            ds_output = wrapper.output
            temp_store = TempStore()
            ds_output.to_zarr(temp_store)
            ds_output.close()
            del ds_output
        finally:
            del wrapper
        return temp_store

    @dask.delayed
    def __call__(
            self,
            ds_initial: xr.Dataset,
            ds_forcing_params: xr.Dataset,
            lead_time: Union[pd.Timedelta, str]
    ) -> "dask.delayed.Delayed":
        temp_store = self._propagate_to_tempstore(
            ds_initial, ds_forcing_params, lead_time
        )
        return temp_store
