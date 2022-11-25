#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 08.02.22
#
# Created for Paper SASIP screening
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}


# System modules
import logging
import subprocess
from typing import Union

# External modules
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

import distributed

# Internal modules


logger = logging.getLogger(__name__)


def get_git_revision_short_hash() -> str:
    return subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']
    ).decode('ascii').strip()


def initialize_cluster_client(
        n_workers: int,
        cluster_address: Union[str, None] = None,
        memory_limit: str = "auto",
) -> distributed.Client:
    """
    Initialize a cluster client.

    Parameters
    ----------
    n_workers : int
        The number of workers.
    memory_limit : str
        The memory limit for the workers. The default is set to auto.

    Returns
    -------
    Client
        The initialized cluster client.
    """
    if cluster_address is None:
        cluster = distributed.LocalCluster(
            n_workers=n_workers, threads_per_worker=1,
            local_directory='/tmp/distributed', memory_limit=memory_limit
        )
        client = distributed.Client(cluster)
        logger.info("Initialised new client")
        logger.info("Dashboard link: {}".format(client.dashboard_link))
    else:
        client = distributed.get_client(cluster_address)
        logger.info(f"Loaded client from {cluster_address:s}")
    return client


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
) -> dict:
    hparams = {}

    hparams["trainer"] = config["trainer"]
    hparams["network"] = config["network"]
    hparams["datamodule"] = config["data"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]
    hparams["optimizer"] = config["optimizer"]

    # Training parameters
    hparams["seed"] = config["seed"]
    hparams["batch_size"] = config["batch_size"]
    hparams["learning_rate"] = config["learning_rate"]

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams["model/params_not_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return hparams
