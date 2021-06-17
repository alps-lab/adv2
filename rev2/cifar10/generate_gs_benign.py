#!/usr/bin/env python
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rev2.cifar10.model_utils import resnet50, CIFAR10_RESNET50_CKPT_PATH
from rev2.cifar10.data_utils import cifar10_normalize
from rev2.gs.generate_gs import generate_gs


def cifar10_resize_postfn(grad):
    grad = grad.abs().max(1, keepdim=True)[0]
    grad = F.avg_pool2d(grad, 2).squeeze(1)
    shape = grad.shape
    grad = grad.view(len(grad), -1)
    grad_min = grad.min(1, keepdim=True)[0]
    grad = grad - grad_min
    grad_max = grad.max(1, keepdim=True)[0]
    grad = grad / torch.max(grad_max, torch.tensor([1e-8], device='cuda'))
    return grad.view(*shape)


def load_model(config):
    pre_fn = cifar10_normalize
    model = resnet50()
    shape = (32, 32)
    nn.DataParallel(model).load_state_dict(torch.load(CIFAR10_RESNET50_CKPT_PATH,
                                                      lambda storage, location: storage)['net'])

    model.train(False)
    if config.device == 'cuda':
        model.cuda()
    return model, pre_fn, shape


def main(config):
    model_tup = load_model(config)

    dobj = np.load(config.data_path)
    img_x, img_y, img_yt = dobj['img_x'], dobj['img_y'], dobj['img_yt']
    benign_gs = generate_gs(model_tup, img_x, img_y, cifar10_resize_postfn, False, batch_size=50)
    save_dobj = {'img_x': img_x, 'img_y': img_y, 'benign_gs': benign_gs, 'img_yt': img_yt}
    np.savez(config.save_path, **save_dobj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('save_path')
    parser.add_argument('-d', '--device', dest='device', choices=['cpu', 'cuda'])
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=50)
    config = parser.parse_args()

    if config.device is None:
        if torch.cuda.is_available():
            config.device = 'cuda'
        else:
            config.device = 'cpu'
    print('configuration:', config)
    main(config)
