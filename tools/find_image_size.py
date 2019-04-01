#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 23:08:50 2019

@author: apple
"""

import os
import argparse

import random

import tqdm
import numpy as np
import pandas as pd


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
    
    results = []
    for i, row in tqdm.tqdm(df_train.iterrows()):
        image_id = row['ImageId']
        image_name = image_id + '.jpg'
        path = os.path.join('./data/ship_train_v2/', image_name)
        size = os.path.getsize(path)
        results.append((image_id, size))
        
    train_pd = pd.DataFrame.from_records(results, columns=['ImageId', 'size'])
    output_filename = os.path.join(raw_images_dir, 'size_train_segmenter.csv')
    train_pd.to_csv(output_filename, index=False)
        
    
    
if __name__ == '__main__':
  main()
