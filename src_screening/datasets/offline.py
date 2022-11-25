#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 07.07.22
#
# Created for Paper SASIP screening
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}


# System modules
import logging

# External modules
import torch
from torch.utils.data import Dataset
import zarr

# Internal modules


logger = logging.getLogger(__name__)


def to_tensor(in_array: "np.ndarray") -> torch.Tensor:
    in_tensor = torch.from_numpy(in_array)
    out_tensor = in_tensor.float()
    return out_tensor


class OfflineDataset(Dataset):
    def __init__(
            self,
            input_path: str,
            target_path: str
    ):
        super().__init__()
        self._input_path = None
        self.input_dataset: zarr.Group = None
        self.input_path = input_path
        self._target_path = None
        self.target_dataset: zarr.Group = None
        self.target_path = target_path

    @property
    def input_path(self) -> str:
        return self._input_path

    @input_path.setter
    def input_path(self, new_path):
        self.input_dataset = zarr.open(new_path, mode='r')
        self._input_path = new_path

    @property
    def target_path(self) -> str:
        return self._target_path

    @target_path.setter
    def target_path(self, new_path):
        self.target_dataset = zarr.open(new_path, mode='r')
        self._target_path = new_path

    def __len__(self):
        return self.input_dataset["faces"].shape[0]

    def __getitem__(self, idx):
        data_dict = {
            "input_nodes": to_tensor(self.input_dataset["nodes"][idx]),
            "input_faces": to_tensor(self.input_dataset["faces"][idx]),
            "error_nodes": to_tensor(self.target_dataset["nodes"][idx]),
            "error_faces": to_tensor(self.target_dataset["faces"][idx]),
        }
        return data_dict
