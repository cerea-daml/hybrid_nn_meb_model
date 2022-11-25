#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 08.07.22
#
# Created for Paper SASIP screening
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import Union

# External modules
import torch.nn
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair

# Internal modules


logger = logging.getLogger(__name__)


class MultiScaleConv2D(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            large_kernel_size: _size_2_t = 3,
            small_kernel_size: _size_2_t = 3,
            stride: _size_2_t = 1,
            large_dilation: _size_2_t = 9,
            small_dilation: _size_2_t = 1,
            small_padding: _size_2_t = 0,
            large_padding: _size_2_t = 0,
            groups: int = 1,
            bias: bool = True,
            device=None,
            dtype=None
    ):
        super().__init__()
        self.small_conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=small_kernel_size,
            stride=stride, dilation=small_dilation, groups=groups,
            bias=bias, device=device, dtype=dtype, padding=small_padding
        )
        self.large_conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=large_kernel_size,
            stride=stride, dilation=large_dilation, groups=groups,
            bias=False, device=device, dtype=dtype, padding=large_padding
        )

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        small_tensor = self.small_conv(in_tensor)
        large_tensor = self.large_conv(in_tensor)
        out_tensor = small_tensor + large_tensor
        return out_tensor
