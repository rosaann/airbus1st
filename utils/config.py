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
  # train
  c.train = edict()
  c.train.dir = './result/out'
  c.train.batch_size = 64
  c.train.num_epochs = 2000
  c.train.num_grad_acc = None

  # evaluation
  c.eval = edict()
  c.eval.batch_size = 64

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
  c.transform = edict()
  c.transform.name = 'default_transform'
  c.transform.num_preprocessor = 4
  c.transform.params = edict()

  # losses
  c.loss = edict()
  c.loss.name = None
  c.loss.params = edict()

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
