from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import glob
import shutil
from collections import defaultdict

import tqdm

import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import imagehash


def get_labels(filenames, df_train, h):
    train_id_str_len = len('050a106a-bbc1-11e8-b2bb-ac1f6b6435d0')
    suffix_len = len('.jpg')

    train_labels = set()
    for filename in filenames:
        id_str = os.path.basename(filename)[:-suffix_len]
        if id_str in df_train.index:
            labels = df_train.loc[id_str]['EncodedPixels']
            train_labels.add(labels)
        


    return list(train_labels)[0]


def find_leak(hash_func, df_train, 
              train_filenames,  test_filenames):
    train_dict = defaultdict(list)
    for filename in tqdm.tqdm(train_filenames):
        image = Image.open(filename)
        h = hash_func(image)
        train_dict[h].append(filename)


    records = []
    for filename in tqdm.tqdm(test_filenames):
        image = Image.open(filename)
        h = hash_func(image)
        if str(h) == '0000000000000000':
            continue

        if h in train_dict:
            labels = get_labels(train_dict[h], df_train, h)
            records.append((os.path.basename(filename[:-len('.jpg')]), labels))

    return pd.DataFrame.from_records(records, columns=['ImageId', 'EncodedPixels'])


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

    test_dir = os.path.join(raw_images_dir, 'ship_test_v2')
    train_dir = os.path.join(raw_images_dir, 'ship_train_v2')

    test_filenames = list(glob.glob(os.path.join(test_dir, '*.jpg')))
    train_filenames = list(glob.glob(os.path.join(train_dir, '*.jpg')))

    hash_func = {'phash': imagehash.phash,
                 'ahash': imagehash.average_hash}

    df_train = pd.read_csv(os.path.join(raw_images_dir, 'train_ship_segmentations_v2.csv'), index_col='ImageId')

    for hash_type, hash_func in hash_func.items():
        df_leak = find_leak(hash_func, df_train,
                            train_filenames,  test_filenames)
        output_filename = os.path.join(data_dir, 'data_leak.{}.csv'.format(hash_type))
        df_leak.to_csv(output_filename, index=False)


if __name__ == '__main__':
    main()
