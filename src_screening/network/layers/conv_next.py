#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 19.08.22
#
# Created for Paper SASIP screening
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}


# System modules
import logging

# External modules
import torch.nn

from hydra.utils import get_class

# Internal modules


logger = logging.getLogger(__name__)


class ConvNextBlock(torch.nn.Module):
    """
    ConvNeXt block heavily inspired by https://arxiv.org/abs/2201.03545 and
    https://huggingface.co/blog/annotated-diffusion
    """
    def __init__(
            self,
            dim: int,
            dim_out: int,
            kernel_size=7,
            mult=2,
            stride=1,
            activation: str = "torch.nn.GELU",
            layer_scale_init_value: float = 1e-6,
            *args, **kwargs
    ):
        super().__init__()
        self.ds_conv = torch.nn.Conv2d(
            dim, dim, kernel_size, groups=dim, stride=stride,
            padding=(kernel_size-1)//2
        )
        self.net = torch.nn.Sequential(
            torch.nn.GroupNorm(1, dim, eps=1E-6),
            torch.nn.Conv2d(dim, dim_out * mult, 1),
            get_class(activation)(),
            torch.nn.Conv2d(dim_out * mult, dim_out, 1),
        )
        self.res_conv = torch.nn.Conv2d(
            dim, dim_out, 1, stride=stride
        ) if dim != dim_out or stride > 1 else None
        self.gamma = torch.nn.Parameter(
            layer_scale_init_value * torch.ones((dim_out, 1, 1)),
            requires_grad=True
        ) if layer_scale_init_value > 0 else None

    def forward(
            self,
            input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        stem_output = self.ds_conv(input_tensor)
        net_output = self.net(stem_output)
        if self.gamma is not None:
            net_output = net_output * self.gamma
        if self.res_conv is not None:
            input_tensor = self.res_conv(input_tensor)
        output_tensor = net_output + input_tensor
        return output_tensor
