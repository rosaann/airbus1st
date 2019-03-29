from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import pandas as pd
import scipy.misc as misc

from torch.utils.data.dataset import Dataset


class DefaultClassifierDataset(Dataset):
    def __init__(self,
                 images_dir,
                 csv_dir,
                 transform=None,
                 idx_fold=0,
                 num_fold=5,
                 
                 **_):

        self.idx_fold = idx_fold
        self.num_fold = num_fold
        self.transform = transform
        self.csv_dir = csv_dir
        self.images_dir = images_dir
        self.load_data()

    def load_data(self):
        df = pd.read_csv(self.csv_dir)
        self.datalist = []
        for _, row in df.iterrows():
            v = row['ImageId']
            img_path = os.path.join(self.images_dir, v )
            ship = 0
            encoder_r = row['EncodedPixels']
            if len(str(encoder_r)) > 1:
                ship = 1
            self.datalist.append({'p':img_path, 's':ship})
        
   

    def __getitem__(self, index):
        example = self.datalist[index]

        filename = example['p']
        image = misc.imread(filename)
        
        ship = example['s']

        if self.transform is not None:
            image = self.transform(image)

        return {'image': image,
                'label': ship
                }

    def __len__(self):
        return self.size


def test():
    dataset = DefaultDataset('data', 'train', None)
    print(len(dataset))
    example = dataset[0]
    example = dataset[1]

    dataset = DefaultDataset('data', 'val', None)
    print(len(dataset))

if __name__ == '__main__':
    test()
