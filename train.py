#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 27.01.22
#
# Created for Paper SASIP screening
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}

# System modules
import logging

# External modules
import hydra
from omegaconf import DictConfig

main_logger = logging.getLogger(__name__)


def train_task(cfg: DictConfig) -> float:
    # Import within main loop to speed up training on jean zay
    from hydra.utils import instantiate
    import wandb
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import LightningLoggerBase, WandbLogger
    from src_screening.utils import log_hyperparameters

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    main_logger.info(f"Instantiating datamodule <{cfg.data._target_}>")
    data_module: pl.LightningDataModule = instantiate(cfg.data)

    main_logger.info(f"Instantiating network <{cfg.network._target_}>")
    model: pl.LightningModule = instantiate(
        cfg.network,
        optimizer_config=cfg.optimizer,
        lr=cfg.learning_rate,
        _recursive_=False
    )
    hydra_params = log_hyperparameters(config=cfg, model=model)
    model.hparams.update(hydra_params)

    if cfg.callbacks is not None:
        callbacks = []
        for _, callback_cfg in cfg.callbacks.items():
            main_logger.info(
                f"Instantiating callback <{callback_cfg._target_}>"
            )
            curr_callback: pl.callbacks.Callback = instantiate(callback_cfg)
            callbacks.append(curr_callback)
    else:
        callbacks = None

    training_logger = None
    if cfg.logger is not None:
        main_logger.info(
            f"Instantiating logger <{cfg.logger._target_}>"
        )
        training_logger: LightningLoggerBase = instantiate(cfg.logger)

    if isinstance(training_logger, WandbLogger):
        main_logger.info("Add hparams to wandb logger")
        training_logger.experiment.config.update(hydra_params)
        if not cfg.sweep:
            main_logger.info("Watch gradients and parameters of model")
            training_logger.watch(model, log="all", log_freq=75)

    main_logger.info(f"Instantiating trainer")
    trainer: pl.Trainer = instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=training_logger
    )

    main_logger.info(f"Starting training")
    trainer.validate(model=model, datamodule=data_module)
    trainer.fit(model=model, datamodule=data_module)

    main_logger.info(f"Training finished")
    val_gaussian = trainer.callback_metrics.get('val/fixed_gaussian')
    val_laplace = trainer.callback_metrics.get('val/fixed_laplace')

    main_logger.info(f"Validation loss: {val_gaussian}; {val_laplace}")
    main_logger.info(
        f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}"
    )
    if isinstance(training_logger, WandbLogger):
        if not cfg.sweep:
            training_logger._scan_and_log_checkpoints(
                trainer.checkpoint_callback
            )
        wandb.finish()
    return val_gaussian


@hydra.main(version_base=None, config_path='configs/', config_name='config')
def main_train(cfg: DictConfig) -> float:
    import numpy as np
    try:
        val_gaussian = train_task(cfg)
    except MemoryError:
        val_gaussian = np.inf
    return val_gaussian


if __name__ == '__main__':
    main_train()
