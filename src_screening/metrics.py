#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 31.01.22
#
# Created for subsinn
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import Union

# External modules
import torch.nn

# Internal modules

logger = logging.getLogger(__name__)


class MeanSquaredError(torch.nn.Module):
    def forward(
            self,
            input_nodes: torch.Tensor,
            input_faces: torch.Tensor,
            target_nodes: torch.Tensor,
            target_faces: torch.Tensor
    ) -> torch.Tensor:
        squared_diff_nodes = (input_nodes - target_nodes) ** 2
        squared_diff_faces = (input_faces - target_faces) ** 2
        mse_nodes = torch.mean(squared_diff_nodes, dim=(0, 2))
        mse_faces = torch.mean(squared_diff_faces, dim=(0, 2))
        total_mse = torch.cat((mse_nodes, mse_faces), dim=-1)
        return total_mse


class MeanAbsoluteError(torch.nn.Module):
    def __init__(self, average_vars: bool = True):
        super().__init__()
        self.average_vars = average_vars

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
        return total_mae


class MeanError(torch.nn.Module):
    def forward(
            self,
            input_nodes: torch.Tensor,
            input_faces: torch.Tensor,
            target_nodes: torch.Tensor,
            target_faces: torch.Tensor
    ) -> torch.Tensor:
        mean_diff_nodes = (input_nodes - target_nodes).mean(dim=(0, 2))
        mean_diff_faces = (input_faces - target_faces).mean(dim=(0, 2))
        total_mean_diff = torch.cat((mean_diff_nodes, mean_diff_faces), dim=-1)
        return total_mean_diff


class PatternCorrelation(torch.nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def estimate_pattern_corr(
            self,
            input_tensor: torch.Tensor,
            target_tensor: torch.Tensor
    ) -> torch.Tensor:
        input_perts = input_tensor-input_tensor.mean(dim=-1, keepdim=True)
        target_perts = target_tensor-target_tensor.mean(dim=-1, keepdim=True)
        cov = (input_perts * target_perts).sum(dim=-1)
        cov = cov / (input_perts.shape[-1] - 1)
        input_std = input_perts.std(dim=-1, unbiased=True)
        target_std = target_perts.std(dim=-1, unbiased=True)
        corr_coeff = cov / (input_std * target_std + self.eps)
        return corr_coeff

    @staticmethod
    def to_z_space(corr_coeff: torch.Tensor) -> torch.Tensor:
        z_transform = torch.arctanh(corr_coeff)
        return z_transform

    @staticmethod
    def from_z_space(z_transform: torch.Tensor) -> torch.Tensor:
        corr_coeff = torch.tanh(z_transform)
        return corr_coeff

    def forward(
            self,
            input_nodes: torch.Tensor,
            input_faces: torch.Tensor,
            target_nodes: torch.Tensor,
            target_faces: torch.Tensor
    ) -> torch.Tensor:
        corr_coeff_nodes = self.estimate_pattern_corr(input_nodes, target_nodes)
        corr_coeff_faces = self.estimate_pattern_corr(input_faces, target_faces)
        corr_coeff = torch.cat((corr_coeff_nodes, corr_coeff_faces), dim=-1)
        z_transform = self.to_z_space(corr_coeff)
        average_z_transform = z_transform.mean(dim=0)
        average_corr_coeff = self.from_z_space(average_z_transform)
        return average_corr_coeff
