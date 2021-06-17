#!/usr/bin/env python
import argparse

import numpy as np
import torch


from expr_attacks.cam_model_def import cam_resnet50, cam_densenet169
from expr_attacks.cam_generator import generate_cams
from expr_attacks.utils import freeze_model


def load_model(config):
    if config.model == 'resnet50':
        model_tup, forward_tup = cam_resnet50()
    if config.model == 'densenet169':
        model_tup, forward_tup = cam_densenet169()
    model = model_tup[0]
    freeze_model(model)
    if config.device == 'gpu':
        model.cuda()
    model.train(False)
    return model_tup, forward_tup


def main(config):
    dobj = np.load(config.data_path)
    dobj_adv = np.load(config.data_path_adv)
    img_yt = dobj['img_yt'].copy()

    cam_config = dict(batch_size=config.batch_size)
    model_tup, forward_tup = load_model(config)
    robj = {}

    img_x = dobj_adv['stadv_x'].copy()
    cam_adv_yt, cam_adv_yt_n = generate_cams(cam_config, model_tup, forward_tup, (img_x, img_yt),
                                                            config.device == 'gpu')
    robj['stadv_cam_yt'] = cam_adv_yt
    robj['stadv_cam_yt_n'] = cam_adv_yt_n
    np.savez(config.save_path, **robj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('data_path_adv')
    parser.add_argument('model', choices=['resnet50', 'densenet169'])
    parser.add_argument('save_path')
    parser.add_argument('-b', '--batch-size', type=int, dest='batch_size', default=40)
    parser.add_argument('--device', choices=['gpu', 'cpu'], dest='device')

    config = parser.parse_args()
    if config.device is None:
        if torch.cuda.is_available():
            config.device = 'gpu'
        else:
            config.device = 'cpu'
    print('Please check the configuration', config)
    main(config)
