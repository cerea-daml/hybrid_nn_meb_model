#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 23.08.22
#
# Created for Paper SASIP screening
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}


# System modules
import logging
import shutil
from typing import Callable, List, Any, Union
import os

# External modules
import xarray as xr
import pandas as pd
from tqdm import tqdm

import dask
from zarr import TempStore


# Internal modules


logger = logging.getLogger(__name__)


class FixedTimeIterator(object):
    def __init__(
            self,
            propagation_func: Callable,
    ):
        self.propagation_func = propagation_func

    def __call__(
            self,
            zarr_store: str,
            ds_initial: xr.Dataset,
            ds_forcing_params: xr.Dataset,
            lead_time: str
    ):
        output_list = []
        for time in tqdm(ds_initial.time.values):
            curr_initial = ds_initial.sel(time=time)
            curr_store = self._propagate_one_step_to_tempstore(
                ds_initial=curr_initial,
                ds_forcing_params=ds_forcing_params,
                lead_time=lead_time
            )
            output_list.append(curr_store)
        logger.info("Finished iterating through time")
        self.store(zarr_store, output_list, concat_dim="time")
        logger.info("Stored all data into the final zarr store")
        return zarr_store

    def _propagate_one_step_to_tempstore(
            self,
            ds_initial: xr.Dataset,
            ds_forcing_params: xr.Dataset,
            lead_time: Union[str, pd.Timedelta] = "1 hour"
    ) -> TempStore:
        ensemble_stores = []
        for ens_mem in ds_initial.ensemble.values:
            ens_initial = ds_initial.sel(ensemble=ens_mem).compute()
            ens_params = ds_forcing_params.sel(ensemble=ens_mem).compute()
            curr_store = self.propagation_func(
                ds_initial=ens_initial,
                ds_forcing_params=ens_params,
                lead_time=lead_time
            )
            ensemble_stores.append(curr_store)
            ens_initial.close()
            ens_params.close()
        logger.info("Submitted tasks for ensemble members")
        ensemble_stores = dask.compute(*ensemble_stores)
        logger.info("Computted the ensemble stores")
        temp_store = TempStore()
        self.store(temp_store, ensemble_stores, concat_dim="ensemble")
        logger.info("Stored the ensemble members into the zarr store")
        return temp_store

    @staticmethod
    def _clean_stores(list_of_stores: List[TempStore]):
        for store in list_of_stores:
            store.rmdir()
            if os.path.isdir(store.path):
                shutil.rmtree(store.path)

    def store(
            self,
            zarr_store: Any,
            list_of_stores: List[Any],
            concat_dim: str = "ensemble"
    ):
        ds_combined = xr.open_mfdataset(
            list_of_stores,
            chunks={
                "time": 1, "ensemble": -1
            },
            concat_dim=concat_dim,
            combine="nested",
            parallel=True,
            engine="zarr"
        )
        ds_combined.to_zarr(zarr_store, mode="w")
        ds_combined.close()
        self._clean_stores(list_of_stores)
