#!/usr/bin/env python
import argparse

import torch
import torch.nn as nn
import numpy as np


from expr_attacks.utils import freeze_model
from rev2.cifar10.mask_generator import generate_masks_v2
from rev2.cifar10.model_utils import resnet50, CIFAR10_RESNET50_CKPT_PATH
from rev2.cifar10.data_utils import cifar10_normalize


def load_model(config):
    pre_fn = cifar10_normalize
    model = resnet50()
    shape = (32, 32)
    nn.DataParallel(model).load_state_dict(torch.load(CIFAR10_RESNET50_CKPT_PATH,
                                                      lambda storage, location: storage)['net'])

    model.train(False)
    if config.device == 'gpu':
        model.cuda()
    return model, pre_fn, shape


def main(config):
    dobj = np.load(config.data_path)
    img_x, img_y, img_yt = dobj['img_x'].copy(), dobj['img_y'].copy(), dobj['img_yt'].copy()
    model_tup = load_model(config)
    freeze_model(model_tup[0])
    mask_config = dict(lr=config.lr, l1_lambda=config.l1_lambda, tv_lambda=config.tv_lambda,
                       noise_std=config.noise_std, n_iters=config.n_iters,
                       batch_size=config.batch_size, verbose=True)
    mask_benign_y = generate_masks_v2(mask_config, model_tup, (img_x, img_y), config.device == 'gpu')
    # mask_benign_yt = generate_masks_v2(mask_config, model_tup, (img_x, img_yt), config.device == 'gpu')
    np.savez(config.save_path, mask_benign_y=mask_benign_y, img_x=img_x, img_yt=img_yt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('save_path')
    parser.add_argument('-b', '--batch-size', type=int, dest='batch_size', default=40)
    parser.add_argument('-n', type=int, dest='n_iters', default=300)
    parser.add_argument('--l1_lambda', type=float, dest='l1_lambda', default=1e-3)
    parser.add_argument('--lr', type=float, dest='lr', default=0.05)
    parser.add_argument('--tv_lambda', type=float, dest='tv_lambda', default=1e-1)
    parser.add_argument('--noise-std', type=float, dest='noise_std', default=0)
    parser.add_argument('--device', choices=['gpu', 'cpu'], dest='device')

    config = parser.parse_args()
    if config.device is None:
        if torch.cuda.is_available():
            config.device = 'gpu'
        else:
            config.device = 'cpu'
    print('Please check the configuration', config)
    main(config)
