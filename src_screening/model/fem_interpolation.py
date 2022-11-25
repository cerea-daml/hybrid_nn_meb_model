#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 25.11.22
#
# Created for Paper SASIP screening
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import Tuple, Union

# External modules
import numpy as np
import xarray as xr
import torch
import pandas as pd

# Internal modules


logger = logging.getLogger(__name__)


def gen_cartesian_coords(
        ds_input: xr.Dataset,
        cartesian_res: Union[None, float] = 2000.,
        target_shape: Union[None, Tuple[int, int]] = None
):
    if cartesian_res is None and target_shape is None:
        raise ValueError("Either cartesian_res or target_shape must be given.")
    bounds = {
        "x": (
            ds_input["Mesh2_node_x"].values.min(),
            ds_input["Mesh2_node_x"].values.max()
        ), "y": (
            ds_input["Mesh2_node_y"].values.min(),
            ds_input["Mesh2_node_y"].values.max()
        )
    }
    if cartesian_res is not None:
        x_range = np.arange(bounds["x"][0], bounds["x"][1]+1, cartesian_res)
        y_range = np.arange(bounds["y"][0], bounds["y"][1]+1, cartesian_res)
    else:
        x_range = np.linspace(bounds["x"][0], bounds["x"][1], target_shape[1])
        y_range = np.linspace(bounds["y"][0], bounds["y"][1], target_shape[0])
    XX, YY = np.meshgrid(x_range, y_range)
    return (XX, YY), (x_range, y_range)


def get_torch_faces_idx(
        ds_input: xr.Dataset,
        cartesian_coords: Tuple[np.ndarray, np.ndarray]
) -> torch.Tensor:
    tri_idx = ds_input.sinn.find_triangle_idx(
        cartesian_coords[0],
        cartesian_coords[1],
    )
    tri_idx = torch.from_numpy(tri_idx).long()
    return tri_idx


def get_torch_nodes_idx(
        ds_input: xr.Dataset,
        cartesian_coords: Tuple[np.ndarray, np.ndarray]
) -> torch.Tensor:
    tri_idx = ds_input.sinn.find_triangle_idx(
        cartesian_coords[0],
        cartesian_coords[1],
    )
    nodes_idx = ds_input["Mesh2_face_nodes"].values[tri_idx]
    nodes_idx = torch.from_numpy(nodes_idx).long()
    return nodes_idx


def estimate_tri_barycenter(
        ds_origin: xr.Dataset, x_cartesian: np.ndarray, y_cartesian: np.ndarray
) -> np.ndarray:
    """
    This method estimates the barycenter of the given cartesian points
    with respect to the triangulation with the given origin dataset.

    Parameters
    ----------
    ds_origin: xr.Dataset
        The barycenteric coordinates with respect to the coordinates and their
        triangulation of this dataset are estimated.
    x_cartesian: np.ndarray
        The cartesian x coordinates of the points. Has as shape (...,).
    y_cartesian: np.ndarray
        The cartesian y coordinates of the points. Has as shape (...,).

    Returns
    -------
    barycenter: np.ndarray
        The barycenter coordinates of the points with respect to the
        triangulation. Has as shape (..., 3).
    """
    tri_idx = ds_origin.sinn.find_triangle_idx(x_cartesian, y_cartesian)
    nodes_idx = ds_origin["Mesh2_face_nodes"].values[tri_idx]
    x_nodes = ds_origin["Mesh2_node_x"].values[nodes_idx]
    y_nodes = ds_origin["Mesh2_node_y"].values[nodes_idx]

    det_mat = (y_nodes[..., 1] - y_nodes[..., 2]) * \
              (x_nodes[..., 0] - x_nodes[..., 2]) + \
              (x_nodes[..., 2] - x_nodes[..., 1]) * \
              (y_nodes[..., 0] - y_nodes[..., 2])

    barycenter_1 = (
        (y_nodes[..., 1] - y_nodes[..., 2]) * (x_cartesian - x_nodes[..., 2]) +
        (x_nodes[..., 2] - x_nodes[..., 1]) * (y_cartesian - y_nodes[..., 2])
    ) / det_mat

    barycenter_2 = (
        (y_nodes[..., 2] - y_nodes[..., 0]) * (x_cartesian - x_nodes[..., 2]) +
        (x_nodes[..., 0] - x_nodes[..., 2]) * (y_cartesian - y_nodes[..., 2])
    ) / det_mat

    barycenter_3 = 1 - barycenter_1 - barycenter_2

    barycenter = np.stack(
        (barycenter_1, barycenter_2, barycenter_3), axis=-1
    )
    return barycenter


class FEMInterpolation(object):
    _name = "Lagrange interpolation"

    def __init__(self, ds_origin: xr.Dataset):
        """
        This interpolation mimics lagrange interpolation as it is used by
        finite element methods. It uses the triangulation of the origin mesh
        for the target mesh. For data defined on the faces, which corresponds to
        finite element zeroth order methods, the constant values of these
        faces are used. For data defined on the nodes, which corresponds to
        finite element first order methods, a barycentric interpolation is used.

        Parameters
        ----------
        ds_origin: xr.Dataset
            The dataset will be interpolated to the mesh of a target dataset.

        Attributes
        ----------
        ds_origin: xr.Dataset
            The dataset will be interpolated to the mesh of a target dataset.

        Methods
        -------
        interpolate(ds_template: xr.Dataset) -> xr.Dataset
            This method interpolates the data from the origin mesh to the
            mesh of the template dataset with Lagrange interpolation.
        """
        super().__init__()
        self._ds_origin = None
        self.ds_origin = ds_origin

    def __call__(self, ds_template: xr.Dataset) -> xr.Dataset:
        ds_interpolated = self.interpolate(ds_template)
        ds_interpolated = ds_interpolated.assign_attrs(
            interpolation="Interpolated with {}".format(self._name)
        )
        return ds_interpolated

    @property
    def ds_origin(self) -> xr.Dataset:
        return self._ds_origin

    @ds_origin.setter
    def ds_origin(self, ds_origin: xr.Dataset):
        if not isinstance(ds_origin, xr.Dataset):
            raise TypeError("ds_template must be of type xr.Dataset")
        if not isinstance(ds_origin.indexes["nMesh2_node"], pd.MultiIndex):
            raise ValueError(
                "ds_template must have a MultiIndex with name 'nMesh2_node'"
            )
        if not isinstance(ds_origin.indexes["nMesh2_face"], pd.MultiIndex):
            raise ValueError(
                "ds_template must have a MultiIndex with name 'nMesh2_face'"
            )
        self._ds_origin = ds_origin.copy()

    def _interpolate_faces(self, ds_template: xr.Dataset) -> xr.Dataset:
        """
        This method interpolates the face data from the origin mesh to the
        template mesh with lagrange interpolation as used by finite element
        zeroth order methods.

        Parameters
        ----------
        ds_template: xr.Dataset
            The dataset containing the target mesh.

        Returns
        -------
        ds_interp_faces: xr.Dataset
            The interpolated dataset with the same variables as the origin
            dataset but with the face information on the template mesh.
        """
        tri_idx = self.ds_origin.sinn.find_triangle_idx(
            ds_template["Mesh2_face_x"].values,
            ds_template["Mesh2_face_y"].values,
        )
        face_vars = [var for var in self.ds_origin.data_vars
                     if "nMesh2_face" in self.ds_origin[var].dims]
        ds_origin_face = self.ds_origin[face_vars].isel(
            nMesh2_face=tri_idx,
        )
        ds_interpolated = ds_origin_face.assign_coords(
            nMesh2_face=ds_template["nMesh2_face"],
        )
        return ds_interpolated

    def _interpolate_nodes(self, ds_template: xr.Dataset) -> xr.Dataset:
        """
        This method interpolates the nodes data from the origin mesh to the
        template mesh with lagrange interpolation as used by finite element
        first order methods.

        Parameters
        ----------
        ds_template: xr.Dataset
            The dataset containing the target mesh.

        Returns
        -------
        ds_interp_nodes: xr.Dataset
            The interpolated dataset with the same variables as the origin
            dataset but with the node information on the template mesh.
        """
        barycenters =estimate_tri_barycenter(
            self.ds_origin,
            ds_template["Mesh2_node_x"].values,
            ds_template["Mesh2_node_y"].values,
        )
        ds_barycenters = xr.DataArray(
            barycenters,
            dims=("new_nodes", "three"),
        )
        tri_idx = self.ds_origin.sinn.find_triangle_idx(
            ds_template["Mesh2_node_x"].values,
            ds_template["Mesh2_node_y"].values,
        )
        nodes_idx = self.ds_origin["Mesh2_face_nodes"].values[tri_idx]
        ds_nodes_idx = xr.DataArray(
            nodes_idx,
            dims=("new_nodes", "three"),
        )
        node_vars = [var for var in self.ds_origin.data_vars
                     if "nMesh2_node" in self.ds_origin[var].dims]
        ds_nodes = self.ds_origin[node_vars]
        expanded_nodes = ds_nodes.isel(nMesh2_node=ds_nodes_idx)
        ds_interp_nodes = (expanded_nodes*ds_barycenters).sum(dim="three")
        ds_interp_nodes = ds_interp_nodes.rename(new_nodes="nMesh2_node")
        ds_interp_nodes = ds_interp_nodes.assign_coords(
            nMesh2_node=ds_template["nMesh2_node"],
        )
        return ds_interp_nodes

    def interpolate(self, ds_template: xr.Dataset) -> xr.Dataset:
        """
        This method interpolates the data from the origin mesh to the target
        mesh with lagrange interpolation as used by finite element methods.

        Parameters
        ----------
        ds_template: xr.Dataset
            The dataset containing the target mesh.

        Returns
        -------
        ds_interpolated: xr.Dataset
            The interpolated dataset with the same variables as the origin
            dataset but with the mesh information of the template dataset.
        """
        ds_interp_nodes = self._interpolate_nodes(ds_template)
        ds_interp_faces = self._interpolate_faces(ds_template)
        ds_interpolated = xr.merge(
            [ds_interp_nodes, ds_interp_faces],
            join="right"
        )
        ds_interpolated = ds_interpolated.assign_coords(
            Mesh2_face_nodes=ds_template["Mesh2_face_nodes"],
        )
        return ds_interpolated
