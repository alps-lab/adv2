#!/usr/bin/env python
import pickle
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
from scipy.spatial.distance import cdist

from expr_detect.commons import LID_MODEL_REGULAR_PGD, LID_MODEL_ACID_RTS, LID_DATA_PATH, LID_BENIGN_DATA_DIR


def extract_lid_features(reference_features, target_features):
    n = len(target_features[0])
    target_features = [f.reshape((n, -1)) for f in target_features]
    reference_features = [f.reshape((len(f), -1)) for f in reference_features]
    dist_mats = [cdist(f1, f2) for f1, f2 in zip(target_features, reference_features)]

    return [np.sort(dist_mat, axis=-1) for dist_mat in dist_mats]


def get_big_batch_features(model_tup, x_np, n_features, cuda):
    batch_size = 40
    features = [[] for i in range(n_features)]
    n = len(x_np)
    n_batches = (n + batch_size - 1) // batch_size
    model, pre_fn = model_tup[:2]
    for i in range(n_batches):
        si = i * batch_size
        ei = min(si + batch_size, n)
        bx_np = x_np[si:ei]
        bx = torch.tensor(bx_np)
        if cuda:
            bx = bx.cuda()
        for j, t in enumerate(model(pre_fn(bx))[1:]):
            features[j].append(t.detach().cpu().numpy())
    features = [np.concatenate(l, axis=0) for l in features]
    return features


def load_pretrained_model(name):
    if name == 'reg_pgd':
        dobj = pickle.load(open(LID_MODEL_REGULAR_PGD, 'rb'))
    elif name == 'acid_rts':
        dobj = pickle.load(open(LID_MODEL_ACID_RTS, 'rb'))
    else:
        raise Exception('invalid name: %s' % name)
    return dobj['scalar'], dobj['model'], dobj['k']


def load_train_data(fold_start=1, fold_end=25, benign_dir=LID_BENIGN_DATA_DIR):
    benign_imgs = []
    for fold_i in range(fold_start, fold_end + 1):
        benign_imgs.append(np.load(os.path.join(benign_dir, LID_DATA_PATH % fold_i))['img_x'])
    benign_imgs = np.concatenate(benign_imgs, axis=0)
    return benign_imgs


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x
