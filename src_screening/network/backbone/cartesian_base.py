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
from typing import Tuple

# External modules
import torch
import torch.nn

from hydra.utils import get_class

# Internal modules
from src_screening.network.layers import ToCartesianLayer, FromCartesianLayer


logger = logging.getLogger(__name__)


class CartesianBase(torch.nn.Module):
    def __init__(
            self,
            cartesian_weights_path: str,
    ):
        """
        Base class for all cartesian backbones.

        Parameters
        ----------
        cartesian_weights_path : str
            Path with the stored and pre-computed cartesian weights.
        """
        super().__init__()
        weights_dict = torch.load(cartesian_weights_path)
        self.to_cartesian = ToCartesianLayer(
            node_weights=weights_dict["node"],
            face_weights=weights_dict["face"]
        )
        self.from_cartesian = FromCartesianLayer(
            node_weights=weights_dict["inv_node"],
            face_weights=weights_dict["inv_face"],
        )
        self.n_features = None

    def get_backbone_prediction(
            self,
            input_tensor: torch.Tensor
    ) -> torch.Tensor:
        pass

    def forward(
            self,
            forecast_nodes: torch.Tensor,
            forecast_faces: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_cartesian = self.to_cartesian(
            forecast_nodes, forecast_faces
        )
        output_cartesian = self.get_backbone_prediction(input_cartesian)
        output_nodes, output_faces = self.from_cartesian(
            output_cartesian
        )
        return output_nodes, output_faces
