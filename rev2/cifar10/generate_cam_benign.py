#!/usr/bin/env python
import argparse

import numpy as np
import torch

from rev2.cifar10.cam_model_def import cam_resnet50
from expr_attacks.utils import freeze_model
from expr_attacks.cam_generator import generate_cams


def load_model(config):
    model_tup, forward_tup = cam_resnet50()
    model = model_tup[0]
    freeze_model(model)
    if config.device == 'cuda':
        model.cuda()
    model.train(False)
    return model_tup, forward_tup


def main(config):
    dobj = np.load(config.data_path)
    img_x, img_y, img_yt = dobj['img_x'].copy(), dobj['img_y'].copy(), dobj['img_yt'].copy()
    cam_config = dict(batch_size=config.batch_size)
    model_tup, forward_tup = load_model(config)
    print(img_x.shape, img_y.shape)
    cams_y, cams_y_n = generate_cams(cam_config, model_tup, forward_tup, (img_x, img_y), cuda=config.device == 'cuda')
    cams_yt, cams_yt_n = generate_cams(cam_config, model_tup, forward_tup, (img_x, img_yt), cuda=config.device == 'cuda')
    np.savez(config.save_path, cam_benign_y=cams_y, cam_benign_y_n=cams_y_n,
             cam_benign_yt=cams_yt, cam_benign_yt_n=cams_yt_n, img_x=img_x, img_y=img_y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('save_path')
    parser.add_argument('-b', '--batch-size', type=int, dest='batch_size', default=50)
    parser.add_argument('--device', choices=['cuda', 'cpu'], dest='device')

    config = parser.parse_args()
    if config.device is None:
        if torch.cuda.is_available():
            config.device = 'cuda'
        else:
            config.device = 'cpu'
    print('Please check the configuration', config)
    main(config)
