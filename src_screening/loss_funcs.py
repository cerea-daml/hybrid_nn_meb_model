#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 21.02.22
#
# Created for Paper SASIP screening
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}


# System modules
import logging
import math
from typing import Iterable

# External modules
import torch

# Internal modules


logger = logging.getLogger(__name__)


class GaussianNLLLoss(torch.nn.Module):
    def __init__(
            self,
            n_vars: int = 9,
            trainable: bool = True,
            scale: Iterable[float] = None
    ):
        super().__init__()
        self.logvar = torch.nn.Parameter(
            torch.zeros(n_vars), requires_grad=trainable
        )
        if scale is not None:
            self.logvar.data[:] = torch.tensor(scale).pow(2).log().to(
                self.logvar
            )
        self.const_part = math.log(math.pi*2)*n_vars/2

    def forward(
            self,
            input_nodes: torch.Tensor,
            input_faces: torch.Tensor,
            target_nodes: torch.Tensor,
            target_faces: torch.Tensor
    ) -> torch.Tensor:
        diff_nodes = (input_nodes - target_nodes).pow(2)
        diff_faces = (input_faces - target_faces).pow(2)
        mse_nodes = torch.mean(diff_nodes, dim=(0, 2))
        mse_faces = torch.mean(diff_faces, dim=(0, 2))
        total_mse = torch.cat((mse_nodes, mse_faces), dim=-1)
        scale = torch.exp(0.5*self.logvar)
        mse_part = torch.sum(total_mse/scale**2)/2
        logvar_part = torch.sum(self.logvar)/2
        nll = mse_part + self.const_part + logvar_part
        return nll


class LaplaceNLLLoss(torch.nn.Module):
    def __init__(
            self,
            n_vars: int = 9,
            trainable: bool = True,
            scale: Iterable[float] = None
    ):
        super().__init__()
        self.log_two_scale = torch.nn.Parameter(
            torch.zeros(n_vars), requires_grad=trainable
        )
        if scale is not None:
            self.log_two_scale.data[:] = torch.log(2*torch.tensor(scale)).to(
                self.log_two_scale
            )

    def forward(
            self,
            input_nodes: torch.Tensor,
            input_faces: torch.Tensor,
            target_nodes: torch.Tensor,
            target_faces: torch.Tensor
    ) -> torch.Tensor:
        abs_diff_nodes = (input_nodes - target_nodes).abs()
        abs_diff_faces = (input_faces - target_faces).abs()
        mae_nodes = torch.mean(abs_diff_nodes, dim=(0, 2))
        mae_faces = torch.mean(abs_diff_faces, dim=(0, 2))
        total_mae = torch.cat((mae_nodes, mae_faces), dim=-1)
        scale = torch.exp(self.log_two_scale) * 0.5
        weighted_mae = torch.sum(total_mae/scale)
        weighted_log_two_scale = torch.sum(self.log_two_scale)
        return weighted_mae + weighted_log_two_scale
