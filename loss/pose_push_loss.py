'''
@author: Tao Wang
@data: 7.22.2021
'''

import torch
# import numpy as np
import torch.nn as nn


def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection/(x_norm * y_norm)
    dist = (1. - dist) / 2
    return dist


class Push_Loss_batch(object):
    '''
    type 1: only compute all batch sim
    '''
    def __init__(self, use_gpu=True):
      
      self.use_gpu = use_gpu
    
    def __call__(self, inputs_1, inputs_2):

        assert inputs_1.shape[1] == inputs_2.shape[1]

        K = inputs_1.shape[0]

        dist = cosine_dist(inputs_1, inputs_2)

        loss = 0

        for i, value in enumerate(dist):

            loss = loss + sum(value[i+1:])

        loss = loss / (K*K)

        return loss


class Push_Loss(object):
    '''
    type 2: only compute one batch sim
    '''
    def __init__(self, use_gpu=True):
      
      self.use_gpu = use_gpu
    
    def __call__(self, inputs_1, inputs_2):

        assert inputs_1.shape[1] == inputs_2.shape[1]

        K = inputs_1.shape[0]

        dist = cosine_dist(inputs_1, inputs_2)

        loss = 0

        for i, value in enumerate(dist):

            loss = loss + value[i]

        loss = loss / K

        return loss



