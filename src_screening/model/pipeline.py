#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 07.09.22
#
# Created for Paper SASIP screening
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}


# System modules
import os
import logging
from typing import Callable, Dict, Tuple
import tempfile

# External modules
import xarray as xr
import torch

# Internal modules
from .combine_functions import combine_normal


logger = logging.getLogger(__name__)


class NeuralNetworkPipeline(object):
    def __init__(
            self,
            network: torch.nn.Module,
            input_climatology: Dict[str, xr.Dataset],
            target_climatology: Dict[str, xr.Dataset],
            combine_function: Callable = combine_normal
    ):
        self.network = network
        self.input_climatology = input_climatology
        self.target_climatology = target_climatology
        self.combine_function = combine_function

    def __call__(
            self,
            ds_initial: xr.Dataset,
            ds_forecast: xr.Dataset
    ) -> xr.Dataset:
        ds_input = self.combine_function(ds_initial, ds_forecast)
        torch_input = self.dataset_to_torch(ds_input)
        torch_output = self.apply_network(*torch_input)
        ds_output = self.convert_to_dataset(torch_output, ds_forecast)
        return ds_output

    def dataset_to_torch(
            self,
            ds_input: xr.Dataset
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_nodes = ds_input.sinn.get_nodes_array()
        input_nodes = input_nodes.stack(channels_1=["time", "var_names"])
        input_nodes = input_nodes.reset_index("channels_1", drop=True)
        input_nodes = input_nodes.transpose("channels_1", "nMesh2_node")
        input_nodes = input_nodes - self.input_climatology["mean"]["nodes"]
        input_nodes = (input_nodes + 1E-8) / \
                      (self.input_climatology["std"]["nodes"] + 1E-8)
        input_nodes = input_nodes.values[None, :, :]
        input_nodes = torch.from_numpy(input_nodes).float()

        input_faces = ds_input.sinn.get_faces_array()
        input_faces = input_faces.stack(channels_2=("time", "var_names"))
        input_faces = input_faces.reset_index("channels_2", drop=True)
        input_faces = input_faces.transpose("channels_2", "nMesh2_face")
        input_faces = input_faces - self.input_climatology["mean"]["faces"]
        input_faces = (input_faces + 1E-8) / \
                      (self.input_climatology["std"]["faces"] + 1E-8)
        input_faces = input_faces.values[None, :, :]
        input_faces = torch.from_numpy(input_faces).float()
        return input_nodes, input_faces

    def apply_network(
            self,
            *input_tensors
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            output_tensors = self.network(*input_tensors)
        return output_tensors

    def convert_to_dataset(
            self,
            network_output: Tuple[torch.Tensor, torch.Tensor],
            ds_forecast: xr.Dataset,
    ):
        ds_forecast = ds_forecast.drop_vars(["velocity_x", "velocity_y"])

        ds_forecast_nodes = ds_forecast.sinn.get_nodes_array()
        ds_forecast_nodes = ds_forecast_nodes.rename(
            {"var_names": "var_names_1"})
        ds_res_nodes = ds_forecast_nodes.copy(
            data=network_output[0][0, :2].detach().numpy()
        )
        ds_res_nodes = ds_res_nodes * self.target_climatology["std"]["nodes"]
        ds_res_nodes = ds_res_nodes + self.target_climatology["mean"]["nodes"]
        ds_corrected_nodes = ds_forecast_nodes + ds_res_nodes
        ds_corrected_nodes = ds_corrected_nodes.to_dataset("var_names_1")

        ds_forecast_faces = ds_forecast.sinn.get_faces_array()
        ds_forecast_faces = ds_forecast_faces.rename(
            {"var_names": "var_names_2"})
        ds_res_faces = ds_forecast_faces.copy(
            data=network_output[1][0, :7].detach().numpy()
        )
        ds_res_faces = ds_res_faces * self.target_climatology["std"]["faces"]
        ds_res_faces = ds_res_faces + self.target_climatology["mean"]["faces"]
        ds_corrected_faces = ds_forecast_faces + ds_res_faces
        ds_corrected_faces = ds_corrected_faces.to_dataset("var_names_2")

        # Empirically correct thickness for values with diffusion
        ds_corrected_faces["thickness"] = ds_corrected_faces["thickness"].where(
            ds_corrected_faces["thickness"] < 1, ds_forecast["thickness"]
        )
        ds_corrected = xr.merge([ds_corrected_nodes, ds_corrected_faces])
        return ds_corrected
