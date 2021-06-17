#!/usr/bin/env python
import argparse

import numpy as np
import torch


from torchvision.models.resnet import resnet50
from expr_attacks.rts_model_resnet50 import RTSResnet50
from expr_attacks.rts_generator import generate_rts, get_default_rts_config
from expr_attacks.mask_generator import generate_masks, get_default_mask_config
from ia_utils.data_utils import imagenet_normalize
from expr_attacks.utils import freeze_model


def test_mask(data_arx, cuda):
    acid_cam_adv_x, acid_cam_succeed = data_arx['acid_cam_adv_x'], data_arx['acid_cam_succeed']
    yt = data_arx['img_yt']
    mask_config = get_default_mask_config()
    model = resnet50(True)
    model.train(False)
    if cuda:
        model.cuda()

    robj = dict(cam_trans_mask=generate_masks(mask_config,
                                              (model, imagenet_normalize, (224, 224)),
                                              (acid_cam_adv_x, yt), cuda))
    return robj


def test_rts(data_arx, cuda):
    acid_cam_adv_x, acid_cam_succeed = data_arx['acid_cam_adv_x'], data_arx['acid_cam_succeed']
    yt = data_arx['img_yt']
    rts_config = get_default_rts_config('resnet50')
    rts_config['batch_size'] = 30
    rts_model = RTSResnet50(rts_config['ckpt_dir'], cuda)
    freeze_model(rts_model.blackbox_model)
    with torch.no_grad():
        robj = dict(cam_trans_rts=generate_rts(rts_config, rts_model, (acid_cam_adv_x, yt), cuda))
    return robj


def test(config):
    data_arx = np.load(config.input_path)
    cuda = config.device == 'gpu'
    dobj = {}
    dobj.update(test_rts(data_arx, cuda))
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
