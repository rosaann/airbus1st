from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .policy_transform import policy_transform
from .tta_transform import tta_transform


def get_transform(transform, split, params=None):
  f = globals().get(transform.name)

  if params is not None:
    return f(split, **params)
  else:
    return f(split, **transform.params)

