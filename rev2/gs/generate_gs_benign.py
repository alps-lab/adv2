#!/usr/bin/env python
import argparse

import numpy as np
import torch

from ia_utils.model_utils import resnet50, densenet169, freeze_model
from ia_utils.data_utils import imagenet_normalize
from rev2.gs.generate_gs import generate_gs, imagenet_resize_postfn


def load_model(config):
    if config.model == 'resnet':
        model = resnet50(True)
    else:
        model = densenet169(True)
    model.to(config.device)
    model.train(False)
    freeze_model(model)
    return model, imagenet_normalize


def main(config):
    model_tup = load_model(config)

    dobj = np.load(config.data_path)
    img_x, img_y, img_yt = dobj['img_x'], dobj['img_y'], dobj['img_yt']
    benign_gs = generate_gs(model_tup, img_x, img_y, imagenet_resize_postfn, False, batch_size=50)
    save_dobj = {'img_x': img_x, 'img_y': img_y, 'benign_gs': benign_gs, 'img_yt': img_yt}
    np.savez(config.save_path, **save_dobj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['resnet', 'densenet'])
    parser.add_argument('data_path')
    parser.add_argument('save_path')
    parser.add_argument('-d', '--device', dest='device', choices=['cpu', 'gpu'])
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=50)
    config = parser.parse_args()

    if config.device is None:
        if torch.cuda.is_available():
            config.device = 'cuda'
        else:
            config.device = 'cpu'
    if config.device == 'gpu':
        config.device = 'cuda'
    print('configuration:', config)
    main(config)
