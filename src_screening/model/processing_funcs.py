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

# External modules
import pandas as pd
import xarray as xr

# Internal modules


logger = logging.getLogger(__name__)


def convert_to_lead_time(ds_output: xr.Dataset) -> xr.Dataset:
    initial_time = ds_output.time.values[0]
    lead_time = ds_output.time.values-initial_time
    ds_output = ds_output.rename({"time": "lead_time"})
    ds_output: xr.Dataset = ds_output.assign_coords(lead_time=lead_time)
    ds_output = ds_output.expand_dims(time=initial_time, axis=1)
    return ds_output
