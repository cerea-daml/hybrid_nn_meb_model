#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 19.11.21
#
# Created for sasip
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2021}  {Tobias Sebastian Finn}


# System modules
import logging

# External modules
import xarray as xr
import numpy as np
import dask
import dask.array as da

# Internal modules


logger = logging.getLogger(__name__)


def to_faces(triangle_nodes: np.ndarray, node_data: np.ndarray) -> np.ndarray:
    """
    Converts the data from nodes to the faces by taking the triangle data and
    estimating the triangle center as average.

    Parameters
    ----------
    triangle_nodes: np.ndarray
        The nodes of the triangles. Should have shape (n_triangles, 3).
    node_data: np.ndarray
        The data of the nodes that will be converted to the faces. Should
        have shape (n_nodes, n_dim).

    Returns
    -------
    face_data: np.ndarray
        The data of the faces, converted from the nodes as average over each
        triangle. Has shape (n_triangles, n_dim).
    """
    triangulated_data = node_data[triangle_nodes]
    face_data = triangulated_data.mean(axis=1)
    return face_data


def post_process_grid(input_dataset: xr.Dataset) -> xr.Dataset:
    """
    Post process the grid dataset. Adds the coordinates for the faces of the
    triangles and stackes the coordinates of the nodes and the faces as
    multi-dimensional arrays.

    Parameters
    ----------
    input_dataset: xr.Dataset
        The input dataset that will be post processed.

    Returns
    -------
    post_processed_dataset: xr.Dataset
        The post processed dataset with the added and stacked coordinates.
    """
    face_x_data = to_faces(
        input_dataset["Mesh2_face_nodes"].values,
        input_dataset["Mesh2_node_x"].values
    )
    face_y_data = to_faces(
        input_dataset["Mesh2_face_nodes"].values,
        input_dataset["Mesh2_node_y"].values
    )
    added_faces_dataset: xr.Dataset = input_dataset.assign(
        Mesh2_face_x=(("nMesh2_face",), face_x_data),
        Mesh2_face_y=(("nMesh2_face",), face_y_data),
    )
    indexed_dataset = added_faces_dataset.set_index(
        nMesh2_face=["Mesh2_face_x", "Mesh2_face_y"],
        nMesh2_node=["Mesh2_node_x", "Mesh2_node_y"]
    )
    post_processed_dataset = indexed_dataset.set_coords([
        "Mesh2_face_nodes", "Mesh2"
    ])
    return post_processed_dataset


def estimate_norm_plane(data, triangles):
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        v1 = data[..., triangles[:, 2], :]-data[..., triangles[:, 0], :]
        v2 = data[..., triangles[:, 1], :]-data[..., triangles[:, 0], :]
    norm_plane = da.stack(
        [
            v1[..., 1] * v2[..., 2] - v1[..., 2] * v2[..., 1],
            v1[..., 2] * v2[..., 0] - v1[..., 0] * v2[..., 2],
            v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0],
        ], axis=-1
    )
    return norm_plane


def estimate_xr_grads(xr_ds):
    u_components = xr.concat(
        [xr_ds["Mesh2_node_x"], xr_ds["Mesh2_node_y"], xr_ds["u"]],
        dim='components'
    )
    u_components["components"] = ["x", "y", "z"]
    u_norm_plane = xr.apply_ufunc(
        estimate_norm_plane,
        u_components,
        input_core_dims=[["nMesh2_node", "components"]],
        output_core_dims=[["nMesh2_face", "components"]],
        dask="allowed",
        kwargs=dict(triangles=xr_ds["Mesh2_face_nodes"].values)
    )

    v_components = xr.concat(
        [xr_ds["Mesh2_node_x"], xr_ds["Mesh2_node_y"], xr_ds["v"]],
        dim='components'
    )
    v_components["components"] = ["x", "y", "z"]
    v_norm_plane = xr.apply_ufunc(
        estimate_norm_plane,
        v_components,
        input_core_dims=[["nMesh2_node", "components"]],
        output_core_dims=[["nMesh2_face", "components"]],
        dask="allowed",
        kwargs=dict(triangles=xr_ds["Mesh2_face_nodes"].values)
    )
    ds_norm_plane = xr.Dataset({"u_grad": u_norm_plane, "v_grad": v_norm_plane})
    ds_grad = xr.concat([
        -ds_norm_plane.sel(components=["x"])/ds_norm_plane.sel(components="z"),
        -ds_norm_plane.sel(components=["y"])/ds_norm_plane.sel(components="z"),
    ], dim="components")
    return ds_grad


def estimate_deform(ds_grad):
    deform_div = ds_grad["u_grad"].sel(components="x") + ds_grad["v_grad"].sel(components="y")
    deform_shear = np.sqrt(
        (
                ds_grad["u_grad"].sel(components="x") - ds_grad["v_grad"].sel(components="y")
        )**2 + (
                ds_grad["u_grad"].sel(components="y") + ds_grad["v_grad"].sel(components="x")
        )**2
    )
    deform_total = np.sqrt(deform_div ** 2 + deform_shear ** 2)
    ds_deform = xr.Dataset({
        "deform_div": deform_div,
        "deform_shear": deform_shear,
        "deform_tot": deform_total
    })
    return ds_deform
