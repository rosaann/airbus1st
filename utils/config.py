from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml
from easydict import EasyDict as edict


def _get_default_config():
  c = edict()

  # dataset
  c.data_classifier = edict()
  c.data_classifier.name = 'DefaultClassifierDataset'
  c.data_classifier.dir = './data'
  c.data_classifier.params = edict()
  
  c.data_segmenter = edict()
  c.data_segmenter.name = 'DefaultSegmenterDataset'
  c.data_segmenter.dir = './data'
  c.data_segmenter.params = edict()

  # model
  c.model_classifier = edict()
  c.model_classifier.name = 'resnet18_classifier'
  c.model_classifier.params = edict()

  c.model_segmenter = edict()
  c.model_segmenter.name = 'resnet18_segmenter'
  c.model_segmenter.params = edict()
  
  c.train = edict()
  c.train.writer_dir = './result/out/writer'
  
  c.train_classifier = edict()
  c.train_classifier.dir = './result/out/classifier/'
  c.train_classifier.batch_size = 64
  c.train_classifier.num_epochs = 2000
  c.train_classifier.num_grad_acc = None
  
  c.train_segmenter = edict()
  c.train_segmenter.dir = './result/out/segmenter/'
  c.train_segmenter.batch_size = 64
  c.train_segmenter.num_epochs = 2000
  c.train_segmenter.num_grad_acc = None

  # evaluation
  c.eval_classifier = edict()
  c.eval_classifier.batch_size = 64
  
  c.eval_segmenter = edict()
  c.eval_segmenter.batch_size = 64

  # optimizer
  c.optimizer_classifier = edict()
  c.optimizer_classifier.name = 'adam'
  c.optimizer_classifier.params = edict()
  
  c.optimizer_segmenter = edict()
  c.optimizer_segmenter.name = 'adam'
  c.optimizer_segmenter.params = edict()

  # scheduler
  c.scheduler = edict()
  c.scheduler.name = 'none'
  c.scheduler.params = edict()

  # transforms
  c.transform_classifier = edict()
  c.transform_classifier.name = 'default_transform'
  c.transform_classifier.num_preprocessor = 4
  c.transform_classifier.params = edict()
  
  c.transform_segmenter = edict()
  c.transform_segmenter.name = 'default_transform'
  c.transform_segmenter.num_preprocessor = 4
  c.transform_segmenter.params = edict()

  # losses
  c.loss_classifier = edict()
  c.loss_classifier.name = None
  c.loss_classifier.params = edict()
  
  c.loss_segmenter = edict()
  c.loss_segmenter.name = None
  c.loss_segmenter.params = edict()

  return c


def _merge_config(src, dst):
  if not isinstance(src, edict):
    return

  for k, v in src.items():
    if isinstance(v, edict):
      _merge_config(src[k], dst[k])
    else:
      dst[k] = v


def load(config_path):
  with open(config_path, 'r') as fid:
    yaml_config = edict(yaml.load(fid))

  config = _get_default_config()
  _merge_config(yaml_config, config)

  return config
