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
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback

import xarray as xr

import matplotlib.pyplot as plt
# Internal modules
from src_screening.model.post_processing import post_process_grid


logger = logging.getLogger(__name__)


class PlotPredictionsCallback(Callback):
    def __init__(self, template_path: str):
        super().__init__()
        ds_template = xr.open_dataset(template_path).isel(time=0)
        ds_template = post_process_grid(ds_template)
        self.triangulation = ds_template.sinn.triangulation
        ds_template.close()

    def get_plots(self, prediction, truth):
        prediction = prediction.cpu().numpy()
        truth = truth.cpu().numpy()
        max_amp = np.abs(truth).max()
        fig, ax = plt.subplots(ncols=2, figsize=(2, 5), dpi=150)

        ax[0].axis('off')
        ax[0].tripcolor(
            self.triangulation, prediction, cmap="coolwarm",
            vmin=-max_amp, vmax=max_amp
        )
        ax[0].set_xlim(-20000, 20000)
        ax[0].set_ylim(-100000, 100000)
        ax[0].text(0.5, 0.98, s="Prediction", ha="center", va="top",
                   transform=ax[0].transAxes)

        ax[1].axis('off')
        ax[1].tripcolor(
            self.triangulation, truth, cmap="coolwarm",
            vmin=-max_amp, vmax=max_amp
        )
        ax[1].set_xlim(-20000, 20000)
        ax[1].set_ylim(-100000, 100000)
        ax[1].text(0.5, 0.98, s="Truth", ha="center", va="top",
                   transform=ax[1].transAxes)
        fig.subplots_adjust(wspace=0.04)
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
            varnames = [
                "u", "v", "stress_xx", "stress_xy", "stress_yy", "damage",
                "cohesion", "area", "thickness"
            ]
            fig_nodes = [
                self.get_plots(
                    outputs[0][0, k], batch["error_nodes"][0, k]
                ) for k in range(batch["error_nodes"].shape[1])
            ]
            fig_faces = [
                self.get_plots(
                    outputs[1][0, k], batch["error_faces"][0, k]
                ) for k in range(batch["error_faces"].shape[1])
            ]
            trainer.logger.log_image(
                f"predictions", fig_nodes+fig_faces, caption=varnames
            )
            plt.close("all")
