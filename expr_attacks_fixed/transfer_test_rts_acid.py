#!/usr/bin/env python
import argparse

import numpy as np
import torch

from torchvision.models.resnet import resnet50
from expr_attacks.mask_generator import generate_masks, get_default_mask_config
from expr_attacks.cam_generator import generate_cams, get_default_cam_config
from expr_attacks.cam_model_def import cam_resnet50
from ia_utils.data_utils import imagenet_normalize


def test_mask(data_arx, cuda):
    acid_rts_adv_x, acid_rts_succeed = data_arx['acid_rts_adv_x'], data_arx['acid_rts_succeed']
    yt = data_arx['img_yt']
    mask_config = get_default_mask_config()
    model = resnet50(True)
    model.train(False)
    if cuda:
        model.cuda()

    robj = dict(rts_trans_mask=generate_masks(mask_config,
                                              (model, imagenet_normalize, (224, 224)),
                                              (acid_rts_adv_x, yt), cuda))
    return robj


def test_cam(data_arx, cuda):
    acid_rts_adv_x, acid_rts_succeed = data_arx['acid_rts_adv_x'], data_arx['acid_rts_succeed']
    yt = data_arx['img_yt']
    cam_config = get_default_cam_config()
    model_tup, forward_fn = cam_resnet50()
    model_tup[0].train(False)
    if cuda:
        model_tup[0].cuda()
    return dict(rts_trans_cam=
                generate_cams(cam_config, model_tup, forward_fn, (acid_rts_adv_x, yt), cuda)[1])


def test(config):
    data_arx = np.load(config.input_path)
    cuda = config.device == 'gpu'
    dobj = {}
    dobj.update(test_cam(data_arx, cuda))
    dobj.update(test_mask(data_arx, cuda))
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
