#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 07.02.22
#
# Created for Paper SASIP screening
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Tuple, Any, List, Dict, Union, Iterable

# External modules
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric

from omegaconf import DictConfig
from hydra.utils import instantiate


# Internal modules
from src_screening import metrics
from src_screening.loss_funcs import LaplaceNLLLoss, GaussianNLLLoss

logger = logging.getLogger(__name__)


class DeterministicOfflineModel(LightningModule):
    def __init__(
            self,
            backbone: DictConfig,
            loss_func: DictConfig,
            optimizer_config: DictConfig,
            scheduler_config: Union[DictConfig, None] = None,
            n_in_channels: int = 20,
            lr: float = 1e-3,
            vars_node: Iterable[str] = ("u", "v"),
            vars_face: Iterable[str] = (
                "stress_xx", "stress_xy", "stress_yy", "damage", "cohesion",
                "area", "thickness"
            )
    ):
        self.included_vars = list(vars_node) + list(vars_face)
        self.loss_func = instantiate(loss_func, n_vars=len(self.included_vars))
        self.eval_losses = {
            'Corr': metrics.PatternCorrelation(),
            'MSE': metrics.MeanSquaredError(),
            'MAE': metrics.MeanAbsoluteError(),
            'Bias': metrics.MeanError(),
        }
        self.fixed_laplace = LaplaceNLLLoss(trainable=False)
        self.fixed_gaussian = GaussianNLLLoss(trainable=False)
        self.best_metric = MinMetric()
        self.lr = lr

        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

        self.backbone = instantiate(
            backbone, n_in_channels=n_in_channels
        )
        self.head_node = torch.nn.Conv1d(
            self.backbone.n_features, len(vars_node), kernel_size=1, bias=True
        )
        self.head_face = torch.nn.Conv1d(
            self.backbone.n_features, len(vars_face), kernel_size=1, bias=True
        )
        self.save_hyperparameters()

    def forward(
            self,
            input_nodes: torch.Tensor,
            input_faces: torch.Tensor,
            *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features_nodes, features_faces = self.backbone(
            input_nodes, input_faces
        )
        output_nodes = self.head_node(features_nodes)
        output_faces = self.head_face(features_faces)
        return output_nodes, output_faces

    def estimate_losses(
            self,
            prediction_nodes: torch.Tensor,
            prediction_faces: torch.Tensor,
            target_nodes: torch.Tensor,
            target_faces: torch.Tensor,
            prefix: str = 'train',
    ) -> torch.Tensor:
        loss_dict = self.loss_func(
            prediction_nodes, prediction_faces,
            target_nodes, target_faces
        )
        if isinstance(loss_dict, torch.Tensor):
            loss_dict = {'loss': loss_dict}
        loss_dict["total_loss"] = torch.sum(
            torch.stack(list(loss_dict.values()))
        )
        for name, loss in loss_dict.items():
            self.log(f'{prefix}/{name}', loss, on_step=False, on_epoch=True,
                     prog_bar=True)
        return loss_dict["total_loss"]

    def training_step(
            self,
            batch_dict: Dict[str, Any],
            batch_idx: int
    ) -> Any:
        prediction_nodes, prediction_faces = self(
            input_nodes=batch_dict["input_nodes"],
            input_faces=batch_dict["input_faces"],
        )
        total_loss = self.estimate_losses(
            prediction_nodes, prediction_faces,
            target_nodes=batch_dict["error_nodes"],
            target_faces=batch_dict["error_faces"],
            prefix='train',
        )
        return total_loss

    def validate_output(
            self,
            batch_dict: Dict[str, Any],
            prefix: str = 'val'
    ) -> Any:
        prediction_nodes, prediction_faces = self(
            input_nodes=batch_dict["input_nodes"],
            input_faces=batch_dict["input_faces"],
        )
        _ = self.estimate_losses(
            prediction_nodes, prediction_faces,
            target_nodes=batch_dict["error_nodes"],
            target_faces=batch_dict["error_faces"],
            prefix=prefix,
        )
        fixed_laplace = self.fixed_laplace(
            prediction_nodes,
            prediction_faces,
            target_nodes=batch_dict["error_nodes"],
            target_faces=batch_dict["error_faces"],
        )
        fixed_gaussian = self.fixed_gaussian(
            prediction_nodes,
            prediction_faces,
            target_nodes=batch_dict["error_nodes"],
            target_faces=batch_dict["error_faces"],
        )
        self.log(
            f'{prefix}/fixed_laplace', fixed_laplace,
            on_step=False, on_epoch=True, prog_bar=False
        )
        self.log(
            f'{prefix}/fixed_gaussian', fixed_gaussian,
            on_step=False, on_epoch=True, prog_bar=True
        )

        for name, loss_func in self.eval_losses.items():
            loss = loss_func(
                prediction_nodes, prediction_faces,
                batch_dict["error_nodes"], batch_dict["error_faces"]
            )
            if loss.dim() > 0:
                for k, l in enumerate(loss):
                    var_name = self.included_vars[k]
                    self.log(
                        f'{prefix}/{name}/{var_name}', l,
                        on_step=False, on_epoch=True, prog_bar=False
                    )
            else:
                self.log(
                    f'{prefix}/{name}', loss, on_step=False, on_epoch=True,
                    prog_bar=False
                )
        return prediction_nodes, prediction_faces, fixed_gaussian

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        return self.validate_output(batch, prefix="val")

    def validation_epoch_end(self, outputs: List[Any]):
        loss_output = [curr_out[2] for curr_out in outputs]
        self.best_metric.update(torch.mean(torch.stack(loss_output)))
        self.log("hp_metric", self.best_metric.compute(), on_epoch=True,
                 prog_bar=False)

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        return self.validate_output(batch, prefix="test")

    def configure_optimizers(
            self
    ) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
        optimizer = instantiate(
            self.optimizer_config,
            params=self.parameters(),
            lr=self.lr,
            _convert_="all"
        )
        if self.scheduler_config is not None:
            scheduler = instantiate(self.scheduler_config, optimizer=optimizer)
            optimizer = {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/fixed_gaussian',
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }
        return optimizer
