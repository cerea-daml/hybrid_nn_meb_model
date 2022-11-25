#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 07.07.22
#
# Created for Paper SASIP screening
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import Tuple

# External modules
import torch

# Internal modules


logger = logging.getLogger(__name__)


class ToCartesianLayer(torch.nn.Module):
    def __init__(
            self,
            node_weights: torch.Tensor,
            face_weights: torch.Tensor
    ):
        super().__init__()
        assert node_weights.dim() == 3, "Node weights need three dimensions"
        assert face_weights.dim() == 3, "Face weights need three dimensions"
        self.register_buffer("node_weights", node_weights.float())
        self.register_buffer("face_weights", face_weights.float())

    def forward(
            self,
            in_node_tensor: torch.Tensor,
            in_face_tensor: torch.Tensor
    ) -> torch.Tensor:
        out_node_tensor = torch.einsum(
            "...cg,ghw->...chw", in_node_tensor, self.node_weights
        )
        out_face_tensor = torch.einsum(
            "...cg,ghw->...chw", in_face_tensor, self.face_weights
        )
        out_tensor = torch.cat([out_node_tensor, out_face_tensor], dim=-3)
        return out_tensor


class FromCartesianLayer(torch.nn.Module):
    def __init__(
            self,
            node_weights: torch.Tensor,
            face_weights: torch.Tensor,
            node_face_slices: Tuple[slice, slice] =
                (slice(None, None), slice(None, None)),
    ):
        super().__init__()
        assert node_weights.dim() == 3, "Node weights need three dimensions"
        assert face_weights.dim() == 3, "Face weights need three dimensions"
        self.register_buffer("node_weights", node_weights.float())
        self.register_buffer("face_weights", face_weights.float())
        self.node_face_slices = node_face_slices

    def forward(
            self,
            input_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out_node_tensor = torch.einsum(
            "...chw,hwg->...cg",
            input_tensor[..., self.node_face_slices[0], :, :],
            self.node_weights
        )
        out_face_tensor = torch.einsum(
            "...chw,hwg->...cg",
            input_tensor[..., self.node_face_slices[1], :, :],
            self.face_weights
        )
        return out_node_tensor, out_face_tensor
