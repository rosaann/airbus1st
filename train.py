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
from utils.checkpoint import *
from utils.metrics import *
from models.model_factory import get_model
from multiprocessing.pool import ThreadPool
from scipy import ndimage
import torch.nn as nn
from tools.gen_gt_images import genBiImage
import torchvision.utils as vutils
import cv2
from torchvision import transforms

def extract_instance_masks_from_binary_mask(args):
    _id, binary_mask = args
    masks = []
    labelled_mask = ndimage.label(binary_mask.detach().cpu().numpy())[0]
    print('labelled_mask shape ', labelled_mask.shape, ' labelled_mask ', labelled_mask)
    labels, areas = np.unique(labelled_mask, return_counts=True)
    print('labels shape ', labels.shape, ' labels ', labels)
    print('areas shape ', areas.shape, ' areas ', areas)
    
    labels = labels[areas >= 80]
    for label in labels:
        if label == 0: continue
        masks.append((_id, labelled_mask == label))
    if len(masks) < 1: return [(_id, None)]
    return masks

def encode_rle(args):
    _id, mask = args
    print('_id ', _id)
    print('mask ', mask)
    if mask is None: return (_id, None)
    print('mask shape ', mask.shape, ' mask ', mask)
    pixels = mask.T.flatten()
    print('mask1 shape ', mask.shape, ' mask ', mask)
    pixels = np.concatenate([[0], pixels, [0]])
    print('pixels shape ', pixels.shape, ' pixels ', pixels)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    print('runs shape ', runs.shape, ' runs ', pixels)
    runs[1::2] -= runs[::2]
    print('runs1 shape ', runs.shape, ' runs ', pixels)
    return (_id, ' '.join(str(x) for x in runs))

def postprocess_segmentation(pool, ids, binary_masks):
   # ids_and_instance_masks = map(extract_instance_masks_from_binary_mask, zip(ids, binary_masks))
   ex_list = []
   for args in  zip(ids, binary_masks):
       ex_list.append(extract_instance_masks_from_binary_mask(args))
       
  # print('ids_and_instance_masks ', len(list(ids_and_instance_masks)))
   s = sum(ex_list, [])
   print('s ', s)
   
   enc_list = []
   for args in s:
       enc_list.append(encode_rle(args))
       
   return enc_list
  # return encode_rle(sum(ids_and_instance_masks, []))

   # return map(encode_rle, sum(ids_and_instance_masks, []))

pool = ThreadPool(2)
def evaluate_segmenter_single_epoch(config, model, dataloader, criterion,
                          epoch, writer, postfix_dict):
    model.eval()

    with torch.no_grad():
        batch_size = config.eval_segmenter.batch_size
        total_size = len(dataloader.dataset)
        total_step = math.ceil(total_size / batch_size)

       # probability_list = []
       # label_list = []
        loss_list = []
        tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)

        for i, data in tbar:
            images = data['image']
            gt = data['gt']
            paths = data['path']
            if torch.cuda.is_available():
                images = images.cuda()
                gt = gt.cuda()
            binary_masks = model(images)
            
            loss = criterion(binary_masks, gt)

            # measure accuracy and record loss
            loss_list.append(loss.item())
            if i == 0:
                
                remaining_ids = list(map(lambda path: path.split('/')[-1], paths))
                #    print('remaining_ids ', remaining_ids)
                results = postprocess_segmentation(pool, remaining_ids[:len(binary_masks)], binary_masks)
                for ir,  encoded_pixels in enumerate( results):
                    transform = transforms.Compose([
                            #transforms.ToPILImage(),
                            transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
                            ])
                 #   print('paths[ir] ', paths[ir])
                    image_src =cv2.imread(paths[ir])
                  #  image_src = transform(image_src)

                    x1 = vutils.make_grid(image_src, normalize=True, scale_each=True)
                    s1 = x1.size()
    
                    if len( list(s1)) >= 2:
                        print('src image ', x1)
                        writer.add_image('result/{}'.format(ir * 2 ), x1, epoch)
                        
                    image_bi =genBiImage(paths[ir], encoded_pixels[1])
                    image_bi = transform(image_bi)

                    x = vutils.make_grid(image_bi, normalize=True, scale_each=True)
                    s = x.size()
       # print(s)
                    if len( list(s)) >= 2:
                        writer.add_image('result/{}'.format(ir * 2 ), x, epoch)

            f_epoch = epoch + i / total_step
            desc = '{:5s}'.format('val')
            desc += ', {:06d}/{:06d}, {:.2f} epoch'.format(i, total_step, f_epoch)
            tbar.set_description(desc)
            tbar.set_postfix(**postfix_dict)

        log_dict = {}
       
        log_dict['loss'] = sum(loss_list) / len(loss_list)

        for key, value in log_dict.items():
            if writer is not None:
                writer.add_scalar('val/{}'.format(key), value, epoch)
            postfix_dict['val/{}'.format(key)] = value

        return 
def train_segmenter_single_epoch(config, model, dataloader, criterion, optimizer,
                       epoch, writer, postfix_dict):
    model.train()

    batch_size = config.train_segmenter.batch_size
    total_size = len(dataloader.dataset)
    total_step = math.ceil(total_size / batch_size)

    log_dict = {}
    tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
    
    total_loss = 0
    for i, data in tbar:
        images = data['image']
        gt = data['gt']
       # paths = data['path']
        
        if torch.cuda.is_available():
            images = images.cuda()
            gt = gt.cuda()
        
        binary_masks = model(images)

        loss = criterion(binary_masks, gt)

            # measure accuracy and record loss
        total_loss += loss.item()

            # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
     #   print('binary_masks ', binary_masks.shape, ' mask ',binary_masks )
    #    remaining_ids = list(map(lambda path: path.split('/')[-1], paths))
    #    print('remaining_ids ', remaining_ids)
    #    results = postprocess_segmentation(pool, remaining_ids[:len(binary_masks)], binary_masks)
      #  print('logits ', logits.shape)
      #  print('labels ', labels.shape)

        if config.train_classifier.num_grad_acc is None:
            optimizer.step()
            optimizer.zero_grad()
        elif (i+1) % config.train_classifier.num_grad_acc == 0:
            optimizer.step()
            optimizer.zero_grad()  

        f_epoch = epoch + i / total_step

        for key, value in log_dict.items():
            postfix_dict['train/{}'.format(key)] = value

        desc = '{:5s}'.format('train')
        desc += ', {:06d}/{:06d}, {:.2f} epoch'.format(i, total_step, f_epoch)
        tbar.set_description(desc)
        tbar.set_postfix(**postfix_dict)

    log_dict['lr'] = optimizer.param_groups[0]['lr']
    
    log_dict['loss'] = total_loss
    if writer is not None:
        for key, value in log_dict.items():
            writer.add_scalar('train/{}'.format(key), value, epoch)
                    
def train_segmenter(config, model, train_dataloader, eval_dataloader, criterion, optimizer, scheduler, writer, start_epoch):
    num_epochs = config.train_segmenter.num_epochs
    
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
    #    train_segmenter_single_epoch(config, model, train_dataloader,
    #                       criterion, optimizer, epoch, writer, postfix_dict)

        # val phase
        evaluate_segmenter_single_epoch(config, model, eval_dataloader,
                                   criterion, epoch, writer, postfix_dict)

      #  if scheduler.name == 'reduce_lr_on_plateau':
      #    scheduler.step(f1)
      #  elif scheduler.name != 'reduce_lr_on_plateau':
      #    scheduler.step()

    #    utils.checkpoint.save_checkpoint(config.train_segmenter.dir, model, optimizer, epoch, 0)

        
    return {'f1': best_f1, 'f1_mavg': best_f1_mavg}
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

    batch_size = config.train_classifier.batch_size
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
      #  print('logits ', logits.shape)
      #  print('labels ', labels.shape)
        loss = criterion(logits, labels.float())
        if aux_logits is not None:
            aux_loss = criterion(aux_logits, labels.float())
            loss = loss + 0.4 * aux_loss
        log_dict['loss'] = loss.item()

        predictions = (probabilities > 0.5).long()
        accuracy = (predictions == labels).sum().float() / float(predictions.numel())
        log_dict['acc'] = accuracy.item()

        loss.backward()

        if config.train_classifier.num_grad_acc is None:
            optimizer.step()
            optimizer.zero_grad()
        elif (i+1) % config.train_classifier.num_grad_acc == 0:
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
        batch_size = config.eval_classifier.batch_size
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
          #  if aux_logits is not None:
          #      aux_loss = criterion(aux_logits, labels.float())
          #      loss = loss + 0.4 * aux_loss
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
        log_dict['f1'] = f1_score(labels, predictions)
        log_dict['loss'] = sum(loss_list) / len(loss_list)

      #  if writer is not None:
      #      for l in range(28):
      #          f1 = f1_score(labels[:,l], predictions[:,l], 'binary')
      #          writer.add_scalar('val/f1_{:02d}'.format(l), f1, epoch)

        for key, value in log_dict.items():
            if writer is not None:
                writer.add_scalar('val/{}'.format(key), value, epoch)
            postfix_dict['val/{}'.format(key)] = value

        return log_dict['f1']
def train_classifier(config, model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, writer, start_epoch):
    num_epochs = config.train_classifier.num_epochs
    
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

      #  if scheduler.name == 'reduce_lr_on_plateau':
      #    scheduler.step(f1)
      #  elif scheduler.name != 'reduce_lr_on_plateau':
      #    scheduler.step()

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
    criterion_classifier = get_loss(config.loss_classifier)
    optimizer_classifier = get_optimizer(config.optimizer_classifier.name, model_classifier.parameters(), config.optimizer_classifier.params)
    optimizer_segmenter = get_optimizer(config.optimizer_segmenter.name, model_segmenter.parameters(), config.optimizer_segmenter.params)


    checkpoint_classifier = get_initial_checkpoint(config.train_classifier.dir)
    if checkpoint_classifier is not None:
        last_epoch, step = load_checkpoint(model_classifier, optimizer_classifier, checkpoint_classifier)
    else:
        last_epoch, step = -1, -1

    print('from classifier checkpoint: {} last epoch:{}'.format(checkpoint_classifier, last_epoch))
    
    ####
    checkpoint_segmenter = get_initial_checkpoint(config.train_segmenter.dir)
    if checkpoint_segmenter is not None:
        last_epoch, step = load_checkpoint(model_segmenter, optimizer_segmenter, checkpoint_segmenter)
    else:
        last_epoch, step = -1, -1

    print('from segmenter checkpoint: {} last epoch:{}'.format(checkpoint_segmenter, last_epoch))
  #  scheduler = get_scheduler(config, optimizer, last_epoch)
  
    writer = SummaryWriter(config.train.dir)
    
    scheduler = 'none'
  #  train_classifier_dataloaders = get_dataloader(config.data_classifier, './data/data_train.csv',config.train_classifier.batch_size, 'train',config.transform_classifier.num_preprocessor, get_transform(config.transform_classifier, 'train'))
  #  eval_classifier_dataloaders = get_dataloader(config.data_classifier, './data/data_val.csv',config.eval_classifier.batch_size, 'val', config.transform_classifier.num_preprocessor, get_transform(config.transform_classifier, 'val'))
  #  test_dataloaders = get_dataloader(config.data_classifier,'./data/data_test.csv', get_transform(config, 'test'))
       
  #  train_classifier(config, model_classifier, train_classifier_dataloaders,eval_classifier_dataloaders, criterion_classifier, optimizer_classifier, scheduler,
  #        writer, last_epoch+1)
    
    criterion_segmenter = nn.NLLLoss()
    
    train_segmenter_dataloaders = get_dataloader(config.data_segmenter, './data/data_train_segmenter.csv',config.train_segmenter.batch_size, 'train',config.transform_segmenter.num_preprocessor, get_transform(config.transform_segmenter, 'train'))
    eval_segmenter_dataloaders = get_dataloader(config.data_segmenter, './data/data_eval_segmenter.csv',config.eval_segmenter.batch_size, 'val', config.transform_segmenter.num_preprocessor, get_transform(config.transform_segmenter, 'val'))
  
    train_segmenter(config, model_segmenter, train_segmenter_dataloaders,eval_segmenter_dataloaders, criterion_segmenter, optimizer_segmenter, scheduler,
          writer, last_epoch+1)
def getSegmenterCriterion():
    hist_path = os.path.join(args.save, 'hist')
    if os.path.isfile(hist_path + '.npy'):
        hist = np.load(hist_path + '.npy')
        print('{}Loaded cached dataset stats{}!!!'.format(CP_Y, CP_C))
    else:
        # Get class weights based on training data
        hist = np.zeros((n_classes), dtype=np.float)
        for batch_idx, (x, yt) in enumerate(data_loader_train):
            h, bins = np.histogram(yt.numpy(), list(range(n_classes + 1)))
            hist += h

        hist = hist/(max(hist))     # Normalize histogram
        print('{}Saving dataset stats{}...'.format(CP_Y, CP_C))
        np.save(hist_path, hist)
        
    criterion_weight = 1/np.log(1.02 + hist)
    criterion_weight[0] = 0
    criterion_segmenter = nn.NLLLoss(Variable(torch.from_numpy(criterion_weight).float().cuda()))
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
