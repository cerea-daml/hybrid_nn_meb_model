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

# External modules
import matplotlib
matplotlib.use('agg')

import numpy as np
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback

import xarray as xr

import matplotlib.pyplot as plt

# Internal modules
from src_screening.model.fem_interpolation import gen_cartesian_coords
from src_screening.model.post_processing import post_process_grid
from src_screening.network.backbone.cartesian_base import CartesianBase


logger = logging.getLogger(__name__)


class PlotFeaturesCallback(Callback):
    def __init__(self, template_path: str, n_cols=16):
        super().__init__()
        ds_template = xr.open_dataset(template_path).isel(time=0)
        ds_template = post_process_grid(ds_template)
        _, cart_coords_xy = gen_cartesian_coords(
            ds_template, cartesian_res=None, target_shape=(128, 32)
        )
        res_x = cart_coords_xy[0][1] - cart_coords_xy[0][0]
        res_y = cart_coords_xy[1][1] - cart_coords_xy[1][0]
        cart_bounds_x = [cart_coords_xy[0][0] - res_x / 2] + list(
            cart_coords_xy[0] + res_x / 2)
        cart_bounds_y = [cart_coords_xy[1][0] - res_y / 2] + list(
            cart_coords_xy[1] + res_y / 2)
        self.plot_bounds = (cart_bounds_x, cart_bounds_y)
        self.triangulation = ds_template.sinn.triangulation
        ds_template.close()
        self.n_cols = n_cols

    def plot_cartesian_features(self, features):
        features = features.cpu().numpy()
        vmax = np.abs(features).max()

        nrows = features.shape[1] // self.n_cols
        figsize = (self.n_cols*0.5, nrows*2)
        fig, ax = plt.subplots(
            nrows=nrows, ncols=self.n_cols, dpi=150, figsize=figsize
        )
        for k, feature in enumerate(features.squeeze(axis=0)):
            vmax = np.abs(feature).max()
            fcol = k % self.n_cols
            frow = k // self.n_cols
            ax[frow, fcol].set_axis_off()
            ax[frow, fcol].pcolormesh(
                *self.plot_bounds, feature,
                cmap="coolwarm", vmin=-vmax, vmax=vmax
            )
            ax[frow, fcol].set_xlim(-20000, 20000)
            ax[frow, fcol].set_ylim(-100000, 100000)
            ax[frow, fcol].text(
                0.5, 0.01, f"{vmax:.2f}",
                ha="center", va="bottom", transform=ax[frow, fcol].transAxes
            )
        fig.subplots_adjust(wspace=0.02, hspace=0.01)
        return fig

    def plot_features_triangular(self, features):
        features = features.cpu().numpy()
        nrows = features.shape[1] // self.n_cols
        figsize = (self.n_cols*0.5, nrows*2)
        fig, ax = plt.subplots(
            nrows=nrows, ncols=self.n_cols, dpi=150, figsize=figsize
        )
        for k, feature in enumerate(features.squeeze(axis=0)):
            vmax = np.abs(feature).max()

            fcol = k % self.n_cols
            frow = k // self.n_cols
            ax[frow, fcol].set_axis_off()
            ax[frow, fcol].tripcolor(
                self.triangulation, feature,
                cmap="coolwarm", vmin=-vmax, vmax=vmax
            )
            ax[frow, fcol].set_xlim(-20000, 20000)
            ax[frow, fcol].set_ylim(-100000, 100000)
            ax[frow, fcol].text(
                0.5, 0.01, f"{vmax:.2f}",
                ha="center", va="bottom", transform=ax[frow, fcol].transAxes
            )
        fig.subplots_adjust(wspace=0.02, hspace=0.01)
        return fig

    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx,
            dataloader_idx
    ):

        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 8 sample image predictions from the first batch
        if batch_idx == 0 and isinstance(trainer.logger, WandbLogger):
            backbone: CartesianBase = trainer.lightning_module.backbone
            with torch.no_grad():
                input_cart = backbone.to_cartesian(
                    batch["input_nodes"][:1], batch["input_faces"][:1]
                )
                features_cart = backbone.get_backbone_prediction(input_cart)
                _, features_faces = backbone.from_cartesian(
                    features_cart
                )
            fig_cartesian = self.plot_cartesian_features(features_cart)
            trainer.logger.log_image(
                "features_cartesian", [fig_cartesian]
            )
            fig_triangular = self.plot_features_triangular(features_faces)
            trainer.logger.log_image(
                "features_triangular", [fig_triangular]
            )
            plt.close("all")
