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
from typing import List

# External modules
import xarray as xr
import dask
from zarr import TempStore

# Internal modules


logger = logging.getLogger(__name__)


def store_single_zarr_store(
        list_of_stores: List[TempStore],
        zarr_store: str,
) -> None:
    open_ = dask.delayed(xr.open_dataset, )
    list_of_ds = [
        open_(
            store,
            decode_cf=False, decode_times=False, decode_timedelta=False,
            engine="zarr",
        ) for store in list_of_stores
    ]
    xr_dataset = dask.delayed(xr.concat)(list_of_ds, dim="ensemble")
    xr_dataset = xr_dataset.compute()
    xr_dataset = xr_dataset.chunk({
        "ensemble": 10,
        "time": 8,
        "nMesh2_face": -1,
        "nMesh2_node": -1
    })
    xr_dataset.to_zarr(zarr_store, mode="w")
    [store.rmdir() for store in list_of_stores]
