#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 10.12.21
#
# Created for subsinn
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2021}  {Tobias Sebastian Finn}


# System modules
import logging
from os import PathLike
from typing import Tuple, Union, Dict, MutableMapping, Mapping, Any, List

# External modules
import xarray as xr
import numpy as np
import torch
import pandas as pd

import matplotlib.tri as mpl_tri

# Internal modules
from .post_processing import post_process_grid, estimate_xr_grads, \
    estimate_deform


logger = logging.getLogger(__name__)


@xr.register_dataset_accessor("sinn")
class SinnAccessor(object):
    def __init__(self, ds: xr.Dataset) -> None:
        self.ds = ds
        self.node_vars = ["u", "v"]
        self.face_vars = [
            "stress_xx", "stress_xy", "stress_yy", "damage", "cohesion",
            "area", "thickness"
        ]
        self._grid_data = None

    @property
    def included_vars(self) -> List[str]:
        return self.node_vars + self.face_vars

    @property
    def n_vars(self) -> int:
        return len(self.included_vars)

    @property
    def n_node_vars(self):
        return len(self.node_vars)

    @property
    def n_face_vars(self):
        return len(self.face_vars)

    def enforce_physical_bounds(self) -> xr.Dataset:
        ds_corrected = self.ds.copy()
        if "damage" in self.ds.data_vars.keys():
            ds_corrected["damage"] = ds_corrected["damage"].clip(
                min=1E-6, max=1
            )
        if "thickness" in self.ds.data_vars.keys():
            ds_corrected["thickness"] = ds_corrected["thickness"].clip(
                min=1E-6
            )
        if "area" in self.ds.data_vars.keys():
            ds_corrected["area"] = ds_corrected["area"].clip(
                min=1E-6, max=1
            )
        if "cohesion" in self.ds.data_vars.keys():
            ds_corrected["cohesion"] = ds_corrected["cohesion"].clip(
                min=1E-6
            )
        if "u" in self.ds.data_vars.keys():
            ds_corrected["u"] = ds_corrected["u"].clip(
                min=-1, max=1
            )
        if "v" in self.ds.data_vars.keys():
            ds_corrected["v"] = ds_corrected["v"].clip(
                min=-1, max=1
            )
        if "stress_xx" in self.ds.data_vars.keys():
            ds_corrected["stress_xx"] = ds_corrected["stress_xx"].clip(
                min=-5E4, max=5E4
            )
        if "stress_xy" in self.ds.data_vars.keys():
            ds_corrected["stress_xy"] = ds_corrected["stress_xy"].clip(
                min=-5E4, max=5E4
            )
        if "stress_yy" in self.ds.data_vars.keys():
            ds_corrected["stress_yy"] = ds_corrected["stress_yy"].clip(
                min=-5E4, max=5E4
            )
        return ds_corrected

    def get_nodes_array(self) -> xr.DataArray:
        """
        Returns the nodes array of this dataset.

        Returns
        -------
        nodes_array: xr.DataArray
            The nodes array of the dataset.
        """
        ds_nodes = self.ds[self.node_vars]
        nodes_array = ds_nodes.to_array("var_names")
        return nodes_array

    def get_faces_array(self) -> xr.DataArray:
        """
        Returns the faces array of this dataset.

        Returns
        -------
        faces_array: xr.DataArray
            The faces array of the dataset.
        """
        ds_faces = self.ds[self.face_vars]
        faces_array = ds_faces.to_array("var_names")
        return faces_array

    def get_saveable_ds(self) -> xr.Dataset:
        """
        Returns a dataset that can be saved to a file. This method resets
        multiindex coordinates into single index coordinates. Add stores the
        additional coordinates as auxiliary coordinates.

        Returns
        -------
        saveable_ds: xr.Dataset
            The dataset that can be saved to a file.
        """
        multi_index_coords = []
        for coord_name, var in self.ds.coords.items():
            try:
                if isinstance(var.to_index(), pd.MultiIndex):
                    multi_index_coords.append(coord_name)
            except ValueError:
                pass
        saveable_ds = self.ds.reset_index(multi_index_coords)
        return saveable_ds

    def convert_mesh_to_coords(self):
        mesh_vars = [
            var_name for var_name in self.ds.data_vars.keys()
            if "Mesh2" in var_name
        ]
        return self.ds.set_coords(mesh_vars)

    def post_process(self) -> xr.Dataset:
        """
        Post processes this dataset. At the moment, it changes the mesh
        coordinates of the returned dataset and adds variables related to the
        deformation of the sea-ice.

        Returns
        -------
        xr.Dataset
            The post processed dataset.
        """
        ds_grads = estimate_xr_grads(self.ds)
        ds_deformation = estimate_deform(ds_grads)
        ds_merged = xr.merge([self.ds, ds_grads, ds_deformation])
        ds_post_processed = post_process_grid(ds_merged)
        return ds_post_processed

    @property
    def triangulation(self) -> mpl_tri.Triangulation:
        """
        Estimates the triangulation of this dataset with its `Mesh2_node_x`
        and `Mesh2_node_y` variables for the coordinates of the nodes and
        `Mesh2_face_nodes` as the indices of the nodes that form the
        triangle.

        Returns
        -------
        mpl_tri.Triangulation
            The triangulation of the dataset.
        """
        nodes = self.ds.Mesh2_node_x.values, self.ds.Mesh2_node_y.values
        faces = self.ds.Mesh2_face_nodes.values
        return mpl_tri.Triangulation(*nodes, triangles=faces)

    def find_triangle_idx(
            self,
            x_values: np.ndarray,
            y_values: np.ndarray
    ) -> np.ndarray:
        """
        Finds the corresponding triangles to the given coordinates with a
        `matplotlib.tri.TriFinder` on the basis of the triangulation of this
        dataset.

        Parameters
        ----------
        x_values : np.ndarray
            The x-coordinates of the points to find the triangles for.
        y_values : np.ndarray
            The y-coordinates of the points to find the triangles for.

        Returns
        -------
        np.ndarray
            The indices of the triangles that contain the given points.
        """
        finder: mpl_tri.TriFinder = self.triangulation.get_trifinder()
        return finder(x_values, y_values)

    def get_torch_data(self, *var_names: Tuple[str]) -> torch.Tensor:
        """
        Returns the data of the given variables as a torch tensor.
        Variables are combined into the last dimension.

        Parameters
        ----------
        *var_names
            The names of the variables to return.

        Returns
        -------
        torch.Tensor
            The data of the given variables as a torch tensor.

        Raises
        ------
        ValueError
            A ValueError is raised if the specified variables have different
            shapes.
        """
        sliced_ds = self.ds[list(var_names)]
        equal_shapes = all([
            sliced_ds[var].shape == sliced_ds[var_names[0]].shape
            for var in var_names
        ])
        if not equal_shapes:
            raise ValueError(
                "The variables have different shapes."
            )
        data_array = sliced_ds.to_array("var_names")
        transposed_array = data_array.transpose(..., "var_names")
        torch_data = torch.from_numpy(transposed_array.values).float()
        return torch_data

    def get_torch_coords(self) -> Dict[str, torch.Tensor]:
        """
        Returns the coordinates of the dataset as a dictionary of torch tensors.

        Returns
        -------
        Dict[str, torch.Tensor]
            The coordinates of the dataset as a dictionary of torch tensors,
            separated for `nodes` and `faces`. The tensors have two
            dimensions with the first dimension being the number of nodes or
            number of faces and the second dimension being two, for the x and y
            coordinates.
        """
        try:
            coord_data = {
                "faces": self.ds.sinn.get_torch_data(
                    "Mesh2_face_x", "Mesh2_face_y"
                ),
                "nodes": self.ds.sinn.get_torch_data(
                    "Mesh2_node_x", "Mesh2_node_y"
                ),
            }
        except KeyError:
            raise ValueError("Please post-process the grid first!")
        return coord_data

    def to_zarr_metadata(
            self,
            zarr_store: Union[MutableMapping, str, PathLike, None] = None
    ) -> Any:
        """
        Stores the metadata from this dataset into a given zarr store without
        saving the values. The values of the coordinates are stored as type
        of metadata.

        Parameters
        ----------
        zarr_store : MutableMapping, str or path-like, optional
            The zarr store to store the metadata into.
        """
        ds_mesh_coords = self.convert_mesh_to_coords()
        ds_saveable = ds_mesh_coords.sinn.get_saveable_ds()
        ds_saveable.to_zarr(zarr_store, mode="w", compute=False)
        return ds_saveable[ds_saveable.coords.keys()].to_zarr(
            store=zarr_store, mode="r+", compute=True
        )

    def to_zarr_region(
            self,
            zarr_store: Union[MutableMapping, str, PathLike, None] = None,
            region: Mapping[str, slice] = None,
            **kwargs
    ) -> Any:
        """
        Stores the data variables into regions in a given zarr store.

        Parameters
        ----------
        zarr_store : MutableMapping, str or path-like, optional
            The zarr store to store the data into.
        region : Mapping[str, slice], optional
            The region to store the data into.
        """
        ds_mesh_coords = self.convert_mesh_to_coords()
        ds_saveable = ds_mesh_coords.sinn.get_saveable_ds()
        ds_saveable = ds_saveable.expand_dims({
            k: [v] for k, v in region.items()
            if k not in ds_saveable.dims
        })
        zarr_vars = xr.open_zarr(zarr_store).variables.keys()
        ds_saveable = ds_saveable.drop_vars([
            var_name for var_name in ds_saveable.variables.keys()
            if var_name not in zarr_vars
        ])
        ds_saveable: xr.Dataset = ds_saveable.drop_vars(
            ds_saveable.coords.keys()
        )
        return ds_saveable.to_zarr(
            zarr_store,
            region=region,
            **kwargs
        )
