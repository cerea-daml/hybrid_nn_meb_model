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
from typing import Tuple
import os
import argparse

# External modules
import xarray as xr
import numpy as np
import torch

# Internal modules
from src_screening.model import fem_interpolation as grid_utils
import utils


logger = logging.getLogger(__name__)


_interim_path = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.realpath(__file__))
        )
    ),
    "data", "interim"
)


parser = argparse.ArgumentParser()
parser.add_argument('--target_shape', nargs='+', type=int)
parser.add_argument('--resolution', type=float)


def estimate_face_weights(
        ds_input: xr.Dataset,
        cartesian_coords: Tuple[np.ndarray, np.ndarray]
) -> torch.Tensor:
    face_idx = grid_utils.get_torch_faces_idx(ds_input, cartesian_coords)
    face_idx = face_idx.view(-1, 1)
    scatter_vals = torch.ones_like(face_idx).float()
    face_weights_mat = torch.zeros(
        face_idx.shape[0], ds_input.nMesh2_face.size
    )
    face_weights_mat.scatter_(1, face_idx, scatter_vals)
    return face_weights_mat


def estimate_node_weights(
        ds_input: xr.Dataset,
        cartesian_coords: Tuple[np.ndarray, np.ndarray]
) -> torch.Tensor:
    node_idx = grid_utils.get_torch_nodes_idx(
        ds_input, cartesian_coords
    )
    node_idx = node_idx.view(-1, 3)
    node_weights = torch.from_numpy(
        grid_utils.estimate_tri_barycenter(
            ds_input, cartesian_coords[0], cartesian_coords[1]
        )
    ).float()
    node_weights = node_weights.view(-1, 3)
    node_weights_mat = torch.zeros(
        node_idx.shape[0], ds_input.nMesh2_node.size
    )
    node_weights_mat.scatter_(1, node_idx, node_weights)
    return node_weights_mat


def estimate_cartesian_weights(
        ds_input: xr.Dataset,
        cartesian_coords: Tuple[np.ndarray, np.ndarray]
) -> Tuple[torch.Tensor, torch.Tensor]:
    node_weights = estimate_node_weights(ds_input, cartesian_coords)
    node_weights = node_weights.view(*cartesian_coords[0].shape, -1)
    node_weights = node_weights.permute(2, 0, 1)
    face_weights = estimate_face_weights(ds_input, cartesian_coords)
    face_weights = face_weights.view(*cartesian_coords[0].shape, -1)
    face_weights = face_weights.permute(2, 0, 1)
    return node_weights, face_weights


def estimate_inverse_weights(
        node_weights: torch.Tensor, face_weights: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    node_weights_view = node_weights.view(node_weights.shape[0], -1)
    inv_node_weights = torch.linalg.pinv(node_weights_view)
    inv_node_weights = inv_node_weights.view(*node_weights.shape[1:], -1)
    face_weights_view = face_weights.view(face_weights.shape[0], -1)
    inv_face_weights = torch.linalg.pinv(face_weights_view)
    inv_face_weights = inv_face_weights.view(*face_weights.shape[1:], -1)
    return inv_node_weights, inv_face_weights


def main(args: argparse.Namespace):
    logger.info(
        f"Starting with preparation of data: got {args.resolution} as "
        f"resolution and {args.target_shape} as target shape."
    )
    template_path = os.path.join(_interim_path, "template_lr.nc")
    ds_template = utils.load_nc_dataset(template_path).isel(time=[0])
    cartesian_coords, _ = grid_utils.gen_cartesian_coords(
        ds_template, args.resolution, args.target_shape
    )
    logger.info("Generated cartesian coordinates")
    cartesian_weights = estimate_cartesian_weights(
        ds_template, cartesian_coords
    )
    logger.info("Generated to cartesian weights")
    inv_weights = estimate_inverse_weights(*cartesian_weights)
    logger.info("Inverted the weights to get from cartesian weights")
    data_dict = {
        "node": cartesian_weights[0],
        "face": cartesian_weights[1],
        "inv_node": inv_weights[0],
        "inv_face": inv_weights[1]
    }
    if args.resolution is not None:
        weight_path = os.path.join(
            _interim_path,
            "cartesian_weights_{0:.0f}.pt".format(args.resolution)
        )
    else:
        weight_path = os.path.join(
            _interim_path,
            "cartesian_weights_{0:d}x{1:d}.pt".format(*args.target_shape)
        )
    torch.save(data_dict, weight_path)
    logger.info(f"Stored the weights at {weight_path:s}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    main(args)
