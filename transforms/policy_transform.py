from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import random

import cv2
import numpy as np

from albumentations import Compose, RandomRotate90, Flip, Transpose, Resize
from albumentations import RandomContrast, RandomBrightness, RandomGamma
from albumentations import Blur, MotionBlur, InvertImg
from albumentations import Rotate, ShiftScaleRotate, RandomScale
from albumentations import GridDistortion, ElasticTransform

from torchvision import transforms


import scipy.misc as misc
POLICIES = [
  ('RandomContrast',    {'limit': (0.1, 0.3), 'p': [0.0, 0.25, 0.5, 0.75]}),
  ('RandomBrightness',  {'limit': (0.1, 0.3), 'p': [0.0, 0.25, 0.5, 0.75]}),
  ('RandomGamma',       {'gamma_limit': ((70, 90), (110, 130)), 'p': [0.0, 0.25, 0.5, 0.75]}),
  ('Blur',              {'blur_limit': (3, 5), 'p': [0.0, 0.25, 0.5, 0.75]}),
  ('MotionBlur',        {'blur_limit': (3, 5), 'p': [0.0, 0.25, 0.5, 0.75]}),
  ('InvertImg',         {'p': [0.0, 0.25, 0.5, 0.75]}),
  ('Rotate',            {'limit': (5, 45), 'p': [0.0, 0.25, 0.5, 0.75]}),
  ('ShiftScaleRotate',  {'shift_limit': (0.03, 0.12), 'scale_limit': 0.0, 'rotate_limit': 0, 'p': [0.0, 0.25, 0.5, 0.75]}),
  ('RandomScale',       {'scale_limit': (0.05, 0.20), 'p': [0.0, 0.25, 0.5, 0.75]}),
  ('GridDistortion',    {'num_steps': (3, 5), 'distort_limit': (0.1, 0.5),  'p': [0.0, 0.25, 0.5, 0.75]}),
  ('ElasticTransform',  {'alpha': 1, 'sigma': (30, 70), 'alpha_affine': (30, 70),  'p': [0.0, 0.25, 0.5, 0.75]}),
]

img_shape = (3,224,224)
def policy_transform(split,
                     policies=None,
                     size=224,
                     per_image_norm=False,
                     mean_std=None,
                     **kwargs):
  means = np.array([127.5, 127.5, 127.5, 127.5])
  stds = np.array([255.0, 255.0, 255.0, 255.0])

  base_aug = Compose([
    RandomRotate90(),
    Flip(),
    Transpose(),
  ])

  if policies is None:
    policies = []

  if isinstance(policies, str):
    with open(policies, 'r') as fid:
      policies = eval(fid.read())
      policies = itertools.chain.from_iterable(policies)

  aug_list = []
  for policy in policies:
    op_1, params_1 = policy[0]
    op_2, params_2 = policy[1]
    print('op_1 ', op_1, ' pa_1 ', params_1)
    print('op_2 ', op_2, ' pa_2 ', params_2)

    aug = Compose([
      globals().get(op_1)(**params_1),
      globals().get(op_2)(**params_2),
    ])
    aug_list.append(aug)

  print('len(aug_list):', len(aug_list))
  resize = Resize(height=size, width=size, always_apply=True)

  def transform(image):
    image = np.array(image)
    if split == 'train':
     # image = base_aug(image=image)['image']
     # if len(aug_list) > 0:
    #     aug = random.choice(aug_list)
    #     image = aug(image=image)['image']
    #  print('image shape ', image.shape)
    #  image = resize(image=image)['image']
   #   image = misc.imresize(image, (size, size)).astype('float32')
      image = cv2.resize(image, (size, size))
      transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
                transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
                ])
    
      image = transform(image)

    else:
    #  if size != image.shape[0]:
      #  image = resize(image=image)['image']
        #image = misc.imresize(image, (size, size)).astype('float32')
      image = cv2.resize(image, (size, size))
      transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
                transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
                ])
    
      image = transform(image)


  #  image = image.astype(np.float32)
  #  if per_image_norm:
  #      mean = np.mean(image.reshape(-1, 3), axis=0)
  #      std = np.std(image.reshape(-1, 3), axis=0)
  #      image -= mean
  #      image /= (std + 0.0000001)
  #  else:
  #      image -= means
  #      image /= stds
  #  image = np.transpose(image, (2, 0, 1))

    return image

  return transform

