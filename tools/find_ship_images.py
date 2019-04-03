#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 17:28:50 2019

@author: zl
"""

import os
import argparse

import random

import tqdm
import numpy as np
import pandas as pd
from math import isnan

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', dest='data_dir',
                        help='the directory of the data',
                        default='', type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir
    raw_images_dir = os.path.join(data_dir, 'data')
    df_train = pd.read_csv(os.path.join(raw_images_dir, 'train_ship_segmentations_v2.csv'))
    
    index_list = []
    for i, row in tqdm.tqdm(df_train.iterrows()):
        encoder = row['EncodedPixels']
        
     #   if encoder is not float('nan'):
        if isinstance(encoder,str):
            if len(encoder) > 1:
              index_list.append(i)
           # print('in ', i, ' e ', encoder)
              continue
        
      #  if len(encoder) >1:
      #      index_list.append(i)
        
    num = len(index_list)
    print('total ', num)
    
    random.shuffle(index_list)
    train_num = int(num * 0.7)
    val_num = int(num * 0.2)
    test_num = num - train_num - val_num
    
    train_data = []
    for i in tqdm.tqdm(range(train_num)):
        img_id = df_train.get_value(index_list[i], 'ImageId')
        encoder_p = df_train.get_value(index_list[i], 'EncodedPixels')
        train_data.append((img_id, encoder_p))
    
    train_pd = pd.DataFrame.from_records(train_data, columns=['ImageId', 'EncodedPixels'])
    output_filename = os.path.join(raw_images_dir, 'data_train_segmenter.csv')
    train_pd.to_csv(output_filename, index=False)
    
    val_data = []
    for i in tqdm.tqdm(range(val_num)):
        img_id = df_train.get_value(index_list[i], 'ImageId')
        encoder_p = df_train.get_value(index_list[i], 'EncodedPixels')
        val_data.append((img_id, encoder_p))
    
    val_pd = pd.DataFrame.from_records(val_data, columns=['ImageId', 'EncodedPixels'])
    output_filename = os.path.join(raw_images_dir, 'data_eval_segmenter.csv')
    val_pd.to_csv(output_filename, index=False)

    test_data = []
    for i in tqdm.tqdm(range(test_num)):
        img_id = df_train.get_value(index_list[i], 'ImageId')
        encoder_p = df_train.get_value(index_list[i], 'EncodedPixels')
        test_data.append((img_id, encoder_p))
    
    test_pd = pd.DataFrame.from_records(test_data, columns=['ImageId', 'EncodedPixels'])
    output_filename = os.path.join(raw_images_dir, 'data_test_segmenter.csv')
    test_pd.to_csv(output_filename, index=False)
    
if __name__ == '__main__':
  main()
