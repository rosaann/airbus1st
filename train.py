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

from datasets.dataset_factory import get_dataloader
from transforms.transform_factory import get_transform
from models.model_factory import get_model
from losses.loss_factory import get_loss
from optimizers.optimizer_factory import get_optimizer
from schedulers.scheduler_factory import get_scheduler
from utils.utils import prepare_train_directories
import utils.config

from models.model_factory import get_model

def inference(model, images):
    logits = model(images)
  #  print('logits ', logits)
    if isinstance(logits, tuple):
        logits, aux_logits = logits
    else:
        aux_logits = None
    probabilities = F.sigmoid(logits)
  #  print('probabilities ', probabilities)
    return logits, aux_logits, probabilities
def train_classifier_single_epoch(config, model, dataloader, criterion, optimizer,
                       epoch, writer, postfix_dict):
    model.train()

    batch_size = config.train.batch_size
    total_size = len(dataloader.dataset)
    total_step = math.ceil(total_size / batch_size)

    log_dict = {}
    tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
    for i, data in tbar:
        images = data['image']
        labels = data['label']
    #    print('images ', images.shape)
   #     print('labels ', labels)
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        logits, aux_logits, probabilities = inference(model, images)
        if aux_logits is not None:
            aux_loss = criterion(aux_logits, labels.float())
            loss = loss + 0.4 * aux_loss
        log_dict['loss'] = loss.item()

        predictions = (probabilities > 0.5).long()
        accuracy = (predictions == labels).sum().float() / float(predictions.numel())
        log_dict['acc'] = accuracy.item()

        loss.backward()

        if config.train.num_grad_acc is None:
            optimizer.step()
            optimizer.zero_grad()
        elif (i+1) % config.train.num_grad_acc == 0:
            optimizer.step()
            optimizer.zero_grad()

        f_epoch = epoch + i / total_step

        log_dict['lr'] = optimizer.param_groups[0]['lr']
        for key, value in log_dict.items():
            postfix_dict['train/{}'.format(key)] = value

        desc = '{:5s}'.format('train')
        desc += ', {:06d}/{:06d}, {:.2f} epoch'.format(i, total_step, f_epoch)
        tbar.set_description(desc)
        tbar.set_postfix(**postfix_dict)

        if i % 100 == 0:
            log_step = int(f_epoch * 10000)
            if writer is not None:
                for key, value in log_dict.items():
                    writer.add_scalar('train/{}'.format(key), value, log_step)
                    
def evaluate_classifier_single_epoch(config, model, dataloader, criterion,
                          epoch, writer, postfix_dict):
    model.eval()

    with torch.no_grad():
        batch_size = config.eval.batch_size
        total_size = len(dataloader.dataset)
        total_step = math.ceil(total_size / batch_size)

        probability_list = []
        label_list = []
        loss_list = []
        tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
        for i, data in tbar:
            images = data['image']
            labels = data['label']
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            logits, aux_logits, probabilities = inference(model, images)

            loss = criterion(logits, labels.float())
            if aux_logits is not None:
                aux_loss = criterion(aux_logits, labels.float())
                loss = loss + 0.4 * aux_loss
            loss_list.append(loss.item())

            probability_list.extend(probabilities.cpu().numpy())
            label_list.extend(labels.cpu().numpy())

            f_epoch = epoch + i / total_step
            desc = '{:5s}'.format('val')
            desc += ', {:06d}/{:06d}, {:.2f} epoch'.format(i, total_step, f_epoch)
            tbar.set_description(desc)
            tbar.set_postfix(**postfix_dict)

        log_dict = {}
        labels = np.array(label_list)
        probabilities = np.array(probability_list)

        predictions = (probabilities > 0.5).astype(int)
        accuracy = np.sum((predictions == labels).astype(float)) / float(predictions.size)

        log_dict['acc'] = accuracy
        log_dict['f1'] = utils.metrics.f1_score(labels, predictions)
        log_dict['loss'] = sum(loss_list) / len(loss_list)

        if writer is not None:
            for l in range(28):
                f1 = utils.metrics.f1_score(labels[:,l], predictions[:,l], 'binary')
                writer.add_scalar('val/f1_{:02d}'.format(l), f1, epoch)

        for key, value in log_dict.items():
            if writer is not None:
                writer.add_scalar('val/{}'.format(key), value, epoch)
            postfix_dict['val/{}'.format(key)] = value

        return f1
def train_classifier(config, model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, writer, start_epoch):
    num_epochs = config.train.num_epochs
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    if torch.cuda.is_available():
        model = model.cuda()

    postfix_dict = {'train/lr': 0.0,
                    'train/acc': 0.0,
                    'train/loss': 0.0,
                    'val/f1': 0.0,
                    'val/acc': 0.0,
                    'val/loss': 0.0}

    f1_list = []
    best_f1 = 0.0
    best_f1_mavg = 0.0
    for epoch in range(start_epoch, num_epochs):
        # train phase
        train_classifier_single_epoch(config, model, train_dataloader,
                           criterion, optimizer, epoch, writer, postfix_dict)

        # val phase
        f1 = evaluate_classifier_single_epoch(config, model, val_dataloader,
                                   criterion, epoch, writer, postfix_dict)

        if config.scheduler.name == 'reduce_lr_on_plateau':
          scheduler.step(f1)
        elif config.scheduler.name != 'reduce_lr_on_plateau':
          scheduler.step()

        utils.checkpoint.save_checkpoint(config.train_classifier.dir, model, optimizer, epoch, 0)

        f1_list.append(f1)
        f1_list = f1_list[-10:]
        f1_mavg = sum(f1_list) / len(f1_list)

        if f1 > best_f1:
            best_f1 = f1
        if f1_mavg > best_f1_mavg:
            best_f1_mavg = f1_mavg
    return {'f1': best_f1, 'f1_mavg': best_f1_mavg}
def run(config):
   # train_dir = config.train.dir
    
    model_classifier = get_model(config.model_classifier.name)
    model_segmenter = get_model(config.model_segmenter.name)
    if torch.cuda.is_available():
        model_classifier = model_classifier.cuda()
        model_segmenter = model_segmenter.cuda()
    criterion = get_loss(config.loss_classifier)
    optimizer_classifier = get_optimizer(config.optimizer_classifier.name, model_classifier.parameters(), config.optimizer_classifier.params)
    optimizer_segmenter = get_optimizer(config.optimizer_segmenter.name, model_segmenter.parameters(), config.optimizer_segmenter.params)


    checkpoint_classifier = utils.checkpoint.get_initial_checkpoint(config.train_classifier.dir)
    if checkpoint is not None:
        last_epoch, step = utils.checkpoint.load_checkpoint(model_classifier, optimizer_classifier, checkpoint_classifier)
    else:
        last_epoch, step = -1, -1

    print('from checkpoint: {} last epoch:{}'.format(checkpoint, last_epoch))
  #  scheduler = get_scheduler(config, optimizer, last_epoch)
    scheduler = 'none'
    train_dataloaders = get_dataloader(config,'./data/data_train.csv', get_transform(config, 'train'))
    val_dataloaders = get_dataloader(config,'./data/data_val.csv', get_transform(config, 'val'))
    test_dataloaders = get_dataloader(config,'./data/data_test.csv', get_transform(config, 'test'))

    

    writer = SummaryWriter(config.train.dir)
    train_classifier(config, model_classifier, train_dataloaders,val_dataloaders, criterion, optimizer_classifier, scheduler,
          writer, last_epoch+1)

def parse_args():
    parser = argparse.ArgumentParser(description='airbus')
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
    prepare_train_directories(config)
    run(config)

    print('success!')


if __name__ == '__main__':
    main()
