import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys

#---------------
#Input
#pred:   (-1, C)
#labels: (-1, C)
#---------------

def simple_dice_loss3D(pred, labels):
    intersect = 2*(pred * labels).sum(0)[1] + 0.02

    ref = pred.pow(2).sum(0)[1] + 0.01
    seg = labels.pow(2).sum(0)[1] + 0.01
    return 1 - ((intersect / (ref + seg)).sum())

#weight_type = 0 no background
#weight_type = 1 uniform
def generalised_dice_loss3D(pred, labels, useGPU, gpu0, weight_type = 'nobackground'):
    num_labels = labels.size()[1]
    num_dv = num_labels
    if weight_type == 'nobackground':
        num_dv -= 1
        weights = np.ones([num_labels], dtype = np.float32)
        weights[0] = 0.0
    elif weight_type == 'uniform':
        weights = np.ones([num_labels], dtype = np.float32)
    elif weight_type == 'square':
        if num_labels != 3:
            raise NotImplementedError()
        weights = np.array([(518803341 + 1496491 + 217508) / (518803341), (518803341 + 1496491 + 217508) / (1496491),(518803341 + 1496491 + 217508) / (217508)])
    else:
        print('Invalid weight num')
        sys.exit()
    if useGPU:
        v = Variable(torch.from_numpy(weights)).cuda(gpu0)
        p = Variable(torch.from_numpy(np.array([0.01], dtype = np.float32))).cuda(gpu0)
        one = Variable(torch.from_numpy(np.array([1], dtype = np.float32))).cuda(gpu0)
        two = Variable(torch.from_numpy(np.array([2], dtype = np.float32))).cuda(gpu0)
        dv = Variable(torch.from_numpy(np.array([1/float(num_dv)], dtype = np.float32))).cuda(gpu0)
    else:
        v = Variable(torch.from_numpy(weights))
        p = Variable(torch.from_numpy(np.array([0.01], dtype = np.float32)))
        one = Variable(torch.from_numpy(np.array([1], dtype = np.float32)))
        two = Variable(torch.from_numpy(np.array([2], dtype = np.float32)))
        dv = Variable(torch.from_numpy(np.array([1/float(num_dv)], dtype = np.float32)))

    intersect = two*(pred * labels).sum(0) + two*p
    ref = pred.pow(2).sum(0) + p
    seg = labels.pow(2).sum(0) + p

    if weight_type == 'square':
        return one - (dv*((intersect / (ref + seg)) * ref.sum(0).pow(2)).sum())
    return one-(dv*((intersect / (ref + seg))*v).sum())