from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import pandas as pd
import scipy.misc as misc
from PIL import Image as pil_image
import cv2

from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch

class DefaultSegmenterDataset(Dataset):
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
        self.images_gt_dir = './data/ship_train_v2_gt/'
        self.load_data()

    def load_data(self):
      #  print('csv_dir ', self.csv_dir)
        df = pd.read_csv(self.csv_dir)
        self.datalist = []
        for _, row in df.iterrows():
            v = row['ImageId']
            img_path = os.path.join(self.images_dir, v )
            gt_path = os.path.join(self.images_gt_dir, (v.split('.')[0] + '.png'))
            
            self.datalist.append({'p':img_path, 'gt':gt_path, 'i':v})
          #  if len(self.datalist) >= 1000:
          #      break
   

    def __getitem__(self, index):
        example = self.datalist[index]

        filename = example['p']
        #image = misc.imread(filename)
        image = cv2.imread(filename )
        
        transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
                transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
                ])
    
        image = transform(image)
        
        gt_path = example['gt']
      #  print('gt_path ', gt_path)
        gt_img = cv2.imread(gt_path)
      #  print('gt_img ', gt_img.shape)
        
        transform_gt = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
                ])
      #  gt_img = gt_img * 255
        gt_img = transform_gt(gt_img)[0]
        print('gt_img2 ', gt_img.shape)
      #  gt_img = torch.sum(gt_img, dim = 0).type(torch.LongTensor)
        gt_img = gt_img.type(torch.LongTensor)
       # print('gt_img3 ', gt_img.shape)

       # if self.transform is not None:
           # print('image_name :', example['i'])
           # image = self.transform(image)

        return {'image': image,
                'gt': gt_img,
                'path':filename
                }

    def __len__(self):
        return len( self.datalist)
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
        print('csv_dir ', self.csv_dir)
        df = pd.read_csv(self.csv_dir)
        self.datalist = []
        for _, row in df.iterrows():
            v = row['ImageId']
            img_path = os.path.join(self.images_dir, v )
            ship = [0]
            encoder_r = row['EncodedPixels']
            if len(str(encoder_r)) > 1:
                ship = [1]
            self.datalist.append({'p':img_path, 's':ship, 'i':v})
           # if len(self.datalist) >= 1000:
           #     break
   

    def __getitem__(self, index):
        example = self.datalist[index]

        filename = example['p']
        #image = misc.imread(filename)
        image = cv2.imread(filename )
        
        ship = example['s']

        if self.transform is not None:
           # print('image_name :', example['i'])
            image = self.transform(image)

        return {'image': image,
                'label': np.array(ship),
                
                }

    def __len__(self):
        return len( self.datalist)


def test():
    dataset = DefaultDataset('data', 'train', None)
    print(len(dataset))
    example = dataset[0]
    example = dataset[1]

    dataset = DefaultDataset('data', 'val', None)
    print(len(dataset))

if __name__ == '__main__':
    test()
