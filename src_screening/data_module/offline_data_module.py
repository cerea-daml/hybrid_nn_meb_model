#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 31.01.22
#
# Created for Paper SASIP screening
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import Optional
import os

# External modules

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

# Internal modules
from ..datasets import OfflineDataset


logger = logging.getLogger(__name__)


__all__ = ['OfflineDataModule']


class OfflineDataModule(LightningDataModule):
    def __init__(
            self,
            data_path: str,
            batch_size: int = 64,
            num_workers: int = 0,
            input_type: str = "normal",
            target_type: str = "normal",
            pin_memory: bool = True,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.input_type = input_type
        self.target_type = target_type

        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = OfflineDataset(
            input_path=os.path.join(
                self.data_path, 'train', 'dataset', f'input_{self.input_type}'
            ),
            target_path=os.path.join(
                self.data_path, 'train', 'dataset', f'target_{self.target_type}'
            ),
        )
        self.eval_dataset = OfflineDataset(
            input_path=os.path.join(
                self.data_path, 'eval', 'dataset', f'input_{self.input_type}'
            ),
            target_path=os.path.join(
                self.data_path, 'eval', 'dataset', f'target_{self.target_type}'
            ),
        )
        self.test_dataset = OfflineDataset(
            input_path=os.path.join(
                self.data_path, 'test', 'dataset', f'input_{self.input_type}'
            ),
            target_path=os.path.join(
                self.data_path, 'test', 'dataset', f'target_{self.target_type}'
            ),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=True,
            batch_size=self.batch_size,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.eval_dataset,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )
