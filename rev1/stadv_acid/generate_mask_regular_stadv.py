#!/usr/bin/env python
import argparse

import torch
import numpy as np
from torchvision.models.resnet import resnet50
from torchvision.models.densenet import densenet169

from ia_utils.data_utils import imagenet_normalize
from expr_attacks.mask_generator import generate_masks_v2
from expr_attacks.utils import freeze_model


def load_model(config):
    pre_fn = imagenet_normalize
    if config.model == 'resnet50':
        model = resnet50(pretrained=True)
        shape = (224, 224)
    if config.model == 'densenet169':
        model = densenet169(pretrained=True)
        shape = (224, 224)
    model.train(False)
    freeze_model(model)
    if config.device == 'gpu':
        model.cuda()
    return model, pre_fn, shape


def main(config):
    dobj = np.load(config.data_path)
    dobj_adv = np.load(config.data_path_adv)
    img_yt = dobj['img_yt'].copy()

    model_tup = load_model(config)
    mask_config = dict(lr=config.lr, l1_lambda=config.l1_lambda, tv_lambda=config.tv_lambda,
                       noise_std=config.noise_std, n_iters=config.n_iters,
                       batch_size=config.batch_size, verbose=True)
    robj = {}
    img_x = dobj_adv['stadv_x'].copy()
    robj['stadv_mask_yt'] = generate_masks_v2(mask_config, model_tup, (img_x, img_yt),
                                           config.device == 'gpu')

    np.savez(config.save_path, **robj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('data_path_adv')
    parser.add_argument('model', choices=['resnet50', 'densenet169'])
    parser.add_argument('save_path')
    parser.add_argument('-b', '--batch-size', type=int, dest='batch_size', default=40)
    parser.add_argument('-n', type=int, dest='n_iters', default=300)
    parser.add_argument('--l1_lambda', type=float, dest='l1_lambda', default=1e-4)
    parser.add_argument('--lr', type=float, dest='lr', default=0.1)
    parser.add_argument('--tv_lambda', type=float, dest='tv_lambda', default=1e-2)
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
