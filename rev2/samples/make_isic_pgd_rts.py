#!/usr/bin/env python
import argparse

import numpy as np
import torch

from expr_attacks.rts_generator import generate_rts, get_default_rts_config
from rev1.isic_acid.rts_model_resnet50 import RTSResnet50, RTS_ISIC_CKPT_PATH
from expr_attacks.utils import freeze_model


def main(config):
    data_arx = np.load(config.adv_data_path)
    adv_x, img_yt = (data_arx['adv_x'].copy(), data_arx['img_yt'].copy())
    rts_config = get_default_rts_config('resnet50')
    rts_config['batch_size'] = config.batch_size
    rts_config['model_confidence'] = 0.

    rts_model = RTSResnet50(RTS_ISIC_CKPT_PATH, config.device == 'gpu')
    freeze_model(rts_model.saliency)
    rts_adv_yt = generate_rts(rts_config, rts_model, (adv_x, img_yt), config.device == 'gpu')

    dobj = dict(data_arx.items())
    dobj['rts_pgd_yt'] =rts_adv_yt
    np.savez(config.save_path, **dobj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('adv_data_path')
    parser.add_argument('save_path')
    parser.add_argument('-b', '--batch-size', type=int, dest='batch_size', default=50)
    parser.add_argument('--device', dest='device', choices=['cpu', 'gpu'])

    config = parser.parse_args()
    if config.device is None:
        if torch.cuda.is_available():
            config.device = 'gpu'
        else:
            config.device = 'cpu'
    print('Please check the configuration', config)
    main(config)
