#!/usr/bin/env python
import argparse
import re

import torch
import numpy as np
from torchvision.models.resnet import resnet50
from torchvision.models.vgg import vgg16
from torchvision.models.densenet import densenet169

from ia_utils.data_utils import imagenet_normalize
from expr_attacks.mask_generator import generate_masks
from expr_attacks.utils import freeze_model


def load_model(config):
    pre_fn = imagenet_normalize
    if config.model == 'vgg16':
        model = vgg16(pretrained=True)
        shape = (224, 224)
    if config.model == 'resnet50':
        model = resnet50(pretrained=True)
        shape = (224, 224)
    if config.model == 'densenet169':
        model = densenet169(pretrained=True)
        shape = (224, 224)
    model.train(False)
    if config.device == 'gpu':
        model.cuda()
    return model, pre_fn, shape


def main(config):
    pattern = re.compile(r'pgd_step_(?P<step>\d+)_adv_x')
    dobj = np.load(config.data_path)
    dobj_adv = np.load(config.data_path_adv)
    keys = list(dobj_adv.keys())
    kept_keys = {}
    for key in keys:
        matchobj = pattern.match(key)
        if matchobj is not None:
            kept_keys[int(matchobj.group('step'))] = key
    img_yt = dobj['img_yt'].copy()

    model_tup = load_model(config)
    freeze_model(model_tup[0])
    mask_config = dict(lr=config.lr, l1_lambda=config.l1_lambda, tv_lambda=config.tv_lambda,
                       noise_std=config.noise_std, n_iters=config.n_iters,
                       batch_size=config.batch_size, verbose=True)
    robj = {}
    for step, key in kept_keys.items():
        img_x = dobj_adv[key].copy()
        robj['pgd_step_%d_adv_mask_yt' % step] = generate_masks(mask_config, model_tup, (img_x, img_yt),
                                                                config.device == 'gpu')

    np.savez(config.save_path, **robj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('data_path_adv')
    parser.add_argument('model', choices=['resnet50', 'vgg16', 'densenet169'])
    parser.add_argument('save_path')
    parser.add_argument('-b', '--batch-size', type=int, dest='batch_size', default=40)
    parser.add_argument('-n', type=int, dest='n_iters', default=500)
    parser.add_argument('--l1_lambda', type=float, dest='l1_lambda', default=0.01)
    parser.add_argument('--lr', type=float, dest='lr', default=0.1)
    parser.add_argument('--tv_lambda', type=float, dest='tv_lambda', default=1e-4)
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
