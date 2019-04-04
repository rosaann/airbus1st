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
        self.load_data()

    def load_data(self):
        print('csv_dir ', self.csv_dir)
        df = pd.read_csv(self.csv_dir)
        self.datalist = []
        for _, row in df.iterrows():
            v = row['ImageId']
            img_path = os.path.join(self.images_dir, v )
            
            encoder_r = row['EncodedPixels']
            image = cv2.imread(img_path )
            shape = image.shape
            h = shape[0]
            w = shape[1]
            mask = np.zeros(w * h)
            encoder = encoder_r
            if v == '4c9da9e4c.jpg':  
                print('id ', v)
          #  print('w ', w)
          #  print('h ', h)
                print('e ', encoder)
            en_list = encoder.split(' ')
            total = w * h
            for i, start in enumerate( en_list):
                if i % 2 == 0:
                   # print('start aaa ', start)
                    num = en_list[i + 1]
                 #   print('num ', num)
                    for n_i in range(int(num)):
                        s= int(start)
                        if v == '4c9da9e4c.jpg':  
                           print('start ', s)
                           print('n_i ', n_i)
                        index = s + n_i 
                        if index < total:
                            mask[s + n_i] = 1
            mask.resize((w, h))
            mask = np.transpose(mask, (1, 0))
            image2 = pil_image.fromarray(mask * 255)
            image2 = image2.convert("1")
            if v == '4c9da9e4c.jpg':   
                path_this = v.split('.')[0] + '.bmp'
                image2.save(path_this )
            self.datalist.append({'p':img_path, 'e':encoder_r, 'i':v})
           # if len(self.datalist) >= 1000:
           #     break
   

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
        
        shape = image.shape
        h = shape[0]
        w = shape[1]
        
        
        
       # if self.transform is not None:
           # print('image_name :', example['i'])
           # image = self.transform(image)

        return {'image': image,
                'mask': mask,
                'name':filename
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
