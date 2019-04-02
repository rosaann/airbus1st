#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:40:14 2019

@author: zl
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
def takeSecond(elem):
    return elem[1]

def main():
    args = parse_args()
    data_dir = args.data_dir
    raw_images_dir = os.path.join(data_dir, 'data')
    df_train = pd.read_csv(os.path.join(raw_images_dir, 'train_ship_segmentations_v2.csv'))
    
    results = []
    for i, row in tqdm.tqdm(df_train.iterrows()):
        image_id = row['ImageId']
        image_name = image_id
        if image_name != '7ea963164.jpg' and image_name != '4add0b9ef.jpg':
            results.append((image_id, row['EncodedPixels']))
        
   # results.sort(key=takeSecond, reverse=True)
    train_pd = pd.DataFrame.from_records(results, columns=['ImageId', 'EncodedPixels'])
    output_filename = os.path.join(raw_images_dir, 'train_ship_segmentations_v2_2.csv')
    train_pd.to_csv(output_filename, index=False)
        
    
    
if __name__ == '__main__':
  main()
