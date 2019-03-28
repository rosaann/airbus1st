#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:30:35 2019

@author: zl
"""
import os
import math
import argparse
import pprint
import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from datasets import get_dataloader
from transforms import get_transform
from models import get_model
from losses.loss_factory import get_loss
from optimizers.optimizer_factory import get_optimizer
from schedulers import get_scheduler
#import utils
import utils.config

from models.model_factory import get_model
from optimizers import get_optimizer
def run(config):
    train_dir = config.train.dir
    
    model_classifier = get_model(config.model_classifier.name)
    model_segmenter = get_model(config.model_segmenter.name)
    if torch.cuda.is_available():
        model_classifier = model_classifier.cuda()
        model_segmenter = model_segmenter.cuda()
    criterion = get_loss(config)
    optimizer_classifier = get_optimizer(config.optimizer_classifier.name, model_classifier.parameters(), config.optimizer_classifier.params)
    optimizer_segmenter = get_optimizer(config.optimizer_segmenter.name, model_segmenter.parameters(), config.optimizer_segmenter.params)


    checkpoint = utils.checkpoint.get_initial_checkpoint(config)
    if checkpoint is not None:
        last_epoch, step = utils.checkpoint.load_checkpoint(model, optimizer, checkpoint)
    else:
        last_epoch, step = -1, -1

    print('from checkpoint: {} last epoch:{}'.format(checkpoint, last_epoch))
    scheduler = get_scheduler(config, optimizer, last_epoch)
    
    dataloaders = {split:get_dataloader(config, split, get_transform(config, split))
                   for split in ['train', 'val']}
    

    writer = SummaryWriter(config.train.dir)
    train(config, model, dataloaders, criterion, optimizer, scheduler,
          writer, last_epoch+1)

def parse_args():
    parser = argparse.ArgumentParser(description='HPA')
    parser.add_argument('--config', dest='config_file',
                        help='configuration filename',
                        default=None, type=str)
    return parser.parse_args()
def main():
    import warnings
    warnings.filterwarnings("ignore")

    print('train airbus Classification Challenge.')
    args = parse_args()
    if args.config_file is None:
      raise Exception('no configuration file')

    config = utils.config.load(args.config_file)
    pprint.PrettyPrinter(indent=2).pprint(config)
    utils.prepare_train_directories(config)
    run(config)

    print('success!')


if __name__ == '__main__':
    main()
