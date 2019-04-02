from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
import torch.utils.data
import torch.utils.data.sampler
from torch.utils.data import DataLoader

from .default import DefaultClassifierDataset
from .small import SmallDataset
from .test import TestDataset


def get_dataset(data, csv_dir, transform=None, last_epoch=-1):
    f = globals().get(data.name)

    return f('./data/ship_train_v2/',
             csv_dir,
             transform=transform,
             **data.params)


def get_dataloader(data,csv_dir, batch_size, split,num_workers, transform=None, **_):
    dataset = get_dataset(data, csv_dir, transform)

    is_train = 'train' == split

    dataloader = DataLoader(dataset,
                            shuffle=is_train,
                            batch_size=batch_size,
                            drop_last=is_train,
                            num_workers=1,
                            pin_memory=False)
    return dataloader
