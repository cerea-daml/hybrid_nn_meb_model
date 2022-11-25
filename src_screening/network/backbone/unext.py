#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 08.03.22
#
# Created for Paper SASIP screening
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import Tuple, Iterable
import math

# External modules
import torch.nn
import numpy as np
from hydra.utils import get_class

# Internal modules
from .cartesian_base import CartesianBase
from ..layers.conv_next import ConvNextBlock


logger = logging.getLogger(__name__)


class DownLayer(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_blocks: int = 2,
            activation: str = "torch.nn.GELU",
            kernel_size: int = 7,
            mult: int = 2
    ):
        super().__init__()
        self.pooling = torch.nn.Sequential(
            torch.nn.GroupNorm(1, in_channels, eps=1E-6),
            torch.nn.Conv2d(
                in_channels, out_channels,
                kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)
            ),
        )
        out_layers = [
            ConvNextBlock(
                dim=out_channels, dim_out=out_channels,
                kernel_size=kernel_size, activation=activation, mult=mult
            )
        ]
        for block in range(1, n_blocks):
            out_layers.append(
                ConvNextBlock(
                    dim=out_channels, dim_out=out_channels,
                    kernel_size=kernel_size, activation=activation, mult=mult
                )
            )
        self.out_layer = torch.nn.Sequential(
            *out_layers
        )

    def forward(
            self,
            in_tensor: torch.Tensor,
    ) -> torch.Tensor:
        pooled_tensor = self.pooling(in_tensor,)
        conv_tensor = self.out_layer(pooled_tensor,)
        return conv_tensor


class UpLayer(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_blocks: int = 2,
            activation: str = "torch.nn.GELU",
            kernel_size: int = 7,
            mult: int = 2
    ):
        super().__init__()
        self.upscaling = torch.nn.Sequential(
            torch.nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True
            ),
            torch.nn.GroupNorm(1, in_channels, eps=1E-6),
            torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)
            )
        )
        out_layers = [
            ConvNextBlock(
                dim=2*out_channels, dim_out=out_channels,
                kernel_size=kernel_size, activation=activation, mult=mult
            )
        ]
        for block in range(1, n_blocks):
            out_layers.append(
                ConvNextBlock(
                    dim=out_channels, dim_out=out_channels,
                    kernel_size=kernel_size, activation=activation, mult=mult
                )
            )
        self.out_layer = torch.nn.Sequential(
            *out_layers
        )

    def forward(
            self,
            in_tensor: torch.Tensor,
            shortcut: torch.Tensor,
    ) -> torch.Tensor:
        upscaled_tensor = self.upscaling(in_tensor)
        combined_tensor = torch.cat([upscaled_tensor, shortcut], dim=1)
        out_tensor = self.out_layer(combined_tensor)
        return out_tensor


class UNextBackbone(CartesianBase):
    def __init__(
            self,
            cartesian_weights_path: str,
            n_in_channels: int,
            n_features: int = 64,
            n_depth: int = 3,
            n_blocks: int = 2,
            mult: int = 2,
            kernel_size: int = 7,
            activation: str = "torch.nn.GELU",
            feature_activation: str = "torch.nn.ReLU"
    ):
        super().__init__(cartesian_weights_path)
        self.n_features = n_features

        self.init_layer = ConvNextBlock(
            n_in_channels, n_features,
            kernel_size=kernel_size,
            activation=activation,
            mult=mult
        )
        self.down_layers, self.up_layers = self._const_net(
            n_features, n_depth, n_blocks,
            kernel_size=kernel_size,
            activation=activation,
            mult=mult
        )
        bottleneck_features = n_features * (2**n_depth)
        self.bottleneck_layer = ConvNextBlock(
            bottleneck_features,
            bottleneck_features,
            kernel_size=kernel_size,
            activation=activation,
            mult=mult
        )
        out_layer = [
            ConvNextBlock(
                n_features, n_features,
                kernel_size=kernel_size,
                activation=activation,
                mult=mult
            ),
        ]
        if feature_activation != "none":
            out_layer.append(get_class(feature_activation)())
        self.out_layer = torch.nn.Sequential(*out_layer)

    @staticmethod
    def _const_net(
            n_features: int = 64,
            n_depth: int = 3,
            n_blocks: int = 2,
            mult: int = 2,
            kernel_size: int = 7,
            activation: str = "torch.nn.GELU",
    ) -> Tuple[torch.nn.ModuleList, torch.nn.ModuleList]:
        down_layers = []
        n_down_feature_list = n_features * (2 ** np.arange(n_depth+1))
        for k, out_features in enumerate(n_down_feature_list[1:]):
            in_features = n_down_feature_list[k]
            down_layers.append(
                DownLayer(
                    in_features,
                    out_features,
                    n_blocks,
                    kernel_size=kernel_size,
                    activation=activation,
                    mult=mult
                )
            )
        up_layers = []
        n_up_feature_list = n_down_feature_list[::-1]
        for k, out_features in enumerate(n_up_feature_list[1:]):
            in_features = n_up_feature_list[k]
            up_layers.append(
                UpLayer(
                    in_features,
                    out_features,
                    n_blocks,
                    kernel_size=kernel_size,
                    activation=activation,
                    mult=mult
                )
            )
        down_layers = torch.nn.ModuleList(down_layers)
        up_layers = torch.nn.ModuleList(up_layers)
        return down_layers, up_layers

    def get_backbone_prediction(
            self, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        init_tensor = self.init_layer(input_tensor)
        down_tensor_list = [init_tensor]
        for layer in self.down_layers:
            down_tensor_list.append(layer(down_tensor_list[-1]))
        down_tensor_list = down_tensor_list[::-1]
        features_tensor = self.bottleneck_layer(down_tensor_list[0])
        for k, layer in enumerate(self.up_layers):
            features_tensor = layer(features_tensor, down_tensor_list[k+1])
        features_tensor = self.out_layer(features_tensor)
        return features_tensor
