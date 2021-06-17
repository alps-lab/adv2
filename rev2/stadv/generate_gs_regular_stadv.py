#!/usr/bin/env python
import argparse

import torch
import numpy as np
from torchvision.models.resnet import resnet50
from torchvision.models.densenet import densenet169

from ia_utils.data_utils import imagenet_normalize
from ia_utils.model_utils import freeze_model
from rev2.gs.generate_gs import generate_gs, imagenet_resize_postfn


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
    if config.device == 'cuda':
        model.cuda()
    return model, pre_fn, shape


def main(config):
    dobj = np.load(config.data_path)
    dobj_adv = np.load(config.data_path_adv)
    img_yt = dobj['img_yt'].copy()
    model_tup = load_model(config)

    robj = {}
    robj['img_x'] = dobj['img_x']
    robj['img_yt'] = img_yt
    stadv_x = dobj_adv['stadv_x'].copy()
    robj['stadv_gs'] = generate_gs(model_tup, stadv_x, img_yt,
                                   imagenet_resize_postfn, False, batch_size=50)
    robj['stadv_x'] = stadv_x
    np.savez(config.save_path, **robj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('data_path_adv')
    parser.add_argument('model', choices=['resnet50', 'densenet169'])
    parser.add_argument('save_path')
    parser.add_argument('-b', '--batch-size', type=int, dest='batch_size', default=40)
    parser.add_argument('--device', choices=['cuda', 'cpu'], dest='device')

    config = parser.parse_args()
    if config.device is None:
        if torch.cuda.is_available():
            config.device = 'cuda'
        else:
            config.device = 'cpu'
    print('Please check the configuration', config)
    main(config)
