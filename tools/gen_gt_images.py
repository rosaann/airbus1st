#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 21:08:26 2019

@author: apple
"""

import os
import argparse

import random

import tqdm
import numpy as np
import pandas as pd
from math import isnan
import cv2
from PIL import Image as pil_image

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
    os.makedirs(os.path.join(raw_images_dir, 'ship_train_v2_gt'), exist_ok=True)
    gt_images_dir = os.path.join(raw_images_dir, 'ship_train_v2_gt')



    df_train = pd.read_csv(os.path.join(raw_images_dir, 'train_ship_segmentations_v2.csv'))
    
    images_dir = './data/ship_train_v2/'
    for i, row in tqdm.tqdm(df_train.iterrows()):
            v = row['ImageId']
            img_path = os.path.join(images_dir, v )
            
            encoder_r = row['EncodedPixels']
            image = cv2.imread(img_path )
            shape = image.shape
            h = shape[0]
            w = shape[1]
            mask = np.zeros(w * h)
            encoder = encoder_r
          #  if v == '4c9da9e4c.jpg':  
          #      print('id ', v)
          #  print('w ', w)
          #  print('h ', h)
          #      print('e ', encoder)
            en_list = encoder.split(' ')
            total = w * h
            for i, start in enumerate( en_list):
                if i % 2 == 0:
                   # print('start aaa ', start)
                    num = en_list[i + 1]
                 #   print('num ', num)
                    for n_i in range(int(num)):
                        s= int(start)
                   #     if v == '4c9da9e4c.jpg':  
                   #        print('start ', s)
                   #        print('n_i ', n_i)
                        index = s + n_i 
                        if index < total:
                            mask[s + n_i] = 1
            mask.resize((w, h))
            mask = np.transpose(mask, (1, 0))
            image2 = pil_image.fromarray(mask * 255)
            image2 = image2.convert("1")
           # if v == '4c9da9e4c.jpg':   
            path_this = v.split('.')[0] + '.png'
            image2.save( os.path.join(gt_images_dir, path_this) )
        
    
    
if __name__ == '__main__':
  main()
