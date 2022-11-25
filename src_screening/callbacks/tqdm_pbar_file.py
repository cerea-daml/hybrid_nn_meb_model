#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 16.08.22
#
# Created for Paper SASIP screening
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}


# System modules
import sys
import logging

from typing import Any

# External modules
from pytorch_lightning.callbacks.progress.tqdm_progress import \
    TQDMProgressBar, Tqdm

# Internal modules


logger = logging.getLogger(__name__)


class FileTQDMProgressBar(TQDMProgressBar):
    def init_train_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for training."""
        bar = Tqdm(
            desc="Training",
            initial=0,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )
        return bar

    def on_sanity_check_start(self, *_) -> None:
        pass

    def on_sanity_check_end(self, *_) -> None:
        pass

    def on_train_start(self, *_: Any) -> None:
        self.main_progress_bar = self.init_train_tqdm()

    def on_train_epoch_start(self, trainer: "pl.Trainer", *_: Any) -> None:
        pass

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", *_: Any) -> None:
        pass

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.main_progress_bar.disable:
            self.main_progress_bar.n += 1
            self.main_progress_bar.set_postfix(
                self.get_metrics(trainer, pl_module),
                refresh=False
            )
            self.main_progress_bar.write("")

    def on_train_end(self, *_: Any) -> None:
        self.main_progress_bar.close()

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_validation_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        pass

    def on_validation_batch_end(self, trainer: "pl.Trainer", *_: Any) -> None:
        pass

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_test_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        pass

    def on_test_batch_end(self, *_: Any) -> None:
        pass

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_predict_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_predict_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        pass

    def on_predict_batch_end(self, *_: Any) -> None:
        pass

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass
