#!/usr/bin/env python
import argparse

import numpy as np
import torch

from expr_attacks.rts_generator import generate_rts, get_default_rts_config
from expr_attacks.rts_model_resnet50 import RTSResnet50
from expr_attacks.rts_model_densenet169 import RTSDensenet169
from expr_attacks.utils import freeze_model


def main(config):
    data_arx = np.load(config.data_path)
    data_adv_arx = np.load(config.data_adv_path)
    img_yt = data_arx['img_yt'].copy()
    rts_config = get_default_rts_config(config.model)
    rts_config['batch_size'] = config.batch_size
    rts_config['ckpt_dir'] = config.ckpt_dir or rts_config['ckpt_dir']

    if config.model == 'resnet50':
        rts_model = RTSResnet50(rts_config['ckpt_dir'], config.device == 'gpu')
    if config.model == 'densenet169':
        rts_model = RTSDensenet169(rts_config['ckpt_dir'], config.device == 'gpu')
    freeze_model(rts_model.saliency)

    robj = {}
    img_x = data_adv_arx['stadv_x'].copy()
    robj['stadv_rts_yt'] = generate_rts(rts_config, rts_model, (img_x, img_yt),
                                        config.device == 'gpu')
    np.savez(config.save_path, **robj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('data_adv_path')
    parser.add_argument('model', choices=['resnet50', 'densenet169'])
    parser.add_argument('save_path')
    parser.add_argument('-b', '--batch-size', type=int, dest='batch_size', default=50)
    parser.add_argument('--device', dest='device', choices=['cpu', 'gpu'])
    parser.add_argument('--ckpt-dir', dest='ckpt_dir')

    config = parser.parse_args()
    if config.device is None:
        if torch.cuda.is_available():
            config.device = 'gpu'
        else:
            config.device = 'cpu'
    print('Please check the configuration', config)
    main(config)
