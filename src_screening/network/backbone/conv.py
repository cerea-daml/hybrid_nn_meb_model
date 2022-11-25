#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 01.03.22
#
# Created for Paper SASIP screening
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import Tuple, Iterable

# External modules
import torch
from hydra.utils import get_class

# Internal modules
from .cartesian_base import CartesianBase
from ..layers import MultiScaleConv2D


logger = logging.getLogger(__name__)


class ConvBackbone(CartesianBase):
    def __init__(
            self,
            cartesian_weights_path: str,
            n_in_channels: int,
            neurons_per_layer: Iterable[int] = (
                    64, 64, 64, 64, 64
            ),
            kernel_size: Tuple[int, int] = (3, 3),
            dilation: Tuple[int, int] = (9, 9),
            activation: str = "torch.nn.GELU",
            feature_activation: str = "torch.nn.ReLU",
            bn: bool = True,
    ):
        super().__init__(cartesian_weights_path)
        self.n_features = neurons_per_layer[-1]
        self.network, self.n_features = self._init_network(
            n_in_channels, neurons_per_layer, kernel_size,
            dilation, activation, feature_activation, bn
        )

    @staticmethod
    def _init_network(
            n_in_channels: int,
            n_hidden: Iterable[int],
            kernel_size: Tuple[int, int],
            dilation: Tuple[int, int],
            activation: str,
            feature_activation: str,
            bn: bool,
    ) -> Tuple[torch.nn.Module, int]:
        layers = []
        curr_channels = n_in_channels
        padding_size = [(dilation[k] * (kernel_size[k] - 1)) // 2
                        for k in range(2)]
        for k, n_channels in enumerate(n_hidden):
            layers.append(
                MultiScaleConv2D(
                    curr_channels, n_channels, large_kernel_size=kernel_size,
                    small_kernel_size=3, large_dilation=dilation,
                    small_dilation=1, bias=~bn, small_padding=1,
                    large_padding=padding_size
                )
            )
            curr_channels = n_channels
            if bn:
                layers.append(torch.nn.BatchNorm2d(curr_channels))
            if activation is not None and k+1 < len(n_hidden):
                layers.append(get_class(activation)())
        layers.append(get_class(feature_activation)())
        return torch.nn.Sequential(*layers), curr_channels

    def get_backbone_prediction(
            self, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        return self.network(input_tensor)
