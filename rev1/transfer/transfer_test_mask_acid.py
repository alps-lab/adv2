#!/usr/bin/env python
import argparse

import numpy as np
import torch

from torchvision.models.resnet import resnet50
from expr_attacks.rts_model_resnet50 import RTSResnet50
from expr_attacks.rts_generator import generate_rts, get_default_rts_config
from expr_attacks.cam_generator import generate_cams, get_default_cam_config
from expr_attacks.cam_model_def import cam_resnet50
from ia_utils.data_utils import imagenet_normalize
from expr_attacks.utils import freeze_model


def test_cam(data_arx, cuda):
    acid_mask_adv_x, acid_mask_succeed = data_arx['acid_mask_adv_x'], data_arx['acid_mask_succeed']
    yt = data_arx['img_yt']
    cam_config = get_default_cam_config()
    model_tup, forward_fn = cam_resnet50()
    model_tup[0].train(False)
    if cuda:
        model_tup[0].cuda()
    return dict(mask_trans_cam=
                generate_cams(cam_config, model_tup, forward_fn, (acid_mask_adv_x, yt), cuda)[1])


def test_rts(data_arx, cuda):
    acid_mask_adv_x, acid_mask_succeed = data_arx['acid_mask_adv_x'], data_arx['acid_mask_succeed']
    yt = data_arx['img_yt']
    rts_config = get_default_rts_config('resnet50')
    rts_config['batch_size'] = 40
    rts_model = RTSResnet50(rts_config['ckpt_dir'], cuda)
    freeze_model(rts_model.blackbox_model)
    with torch.no_grad():
        robj = dict(mask_trans_rts=generate_rts(rts_config, rts_model, (acid_mask_adv_x, yt), cuda))
    return robj


def test(config):
    data_arx = np.load(config.input_path)
    cuda = config.device == 'gpu'
    dobj = {}
    dobj.update(test_cam(data_arx, cuda))
    dobj.update(test_rts(data_arx, cuda))
    np.savez(config.save_path, **dobj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('save_path')
    parser.add_argument('--device', choices=['cpu', 'gpu'], dest='device')

    config = parser.parse_args()
    if config.device is None:
        if torch.cuda.is_available():
            config.device = 'gpu'
        else:
            config.device = 'cpu'
    print('Please check the configuration', config)
    test(config)
