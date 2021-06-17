#!/usr/bin/env python
import argparse

import numpy as np
import torch


from torchvision.models.resnet import resnet50
from expr_attacks.rts_model_resnet50 import RTSResnet50
from expr_attacks.rts_generator import generate_rts, get_default_rts_config
from expr_attacks.mask_generator import generate_masks_v2, get_default_mask_config
from ia_utils.data_utils import imagenet_normalize
from expr_attacks.utils import freeze_model
from expr_attacks.cam_generator import generate_cams, get_default_cam_config
from expr_attacks.cam_model_def import cam_resnet50


def test_mask(data_arx, cuda):
    acid_gs_adv_x, acid_gs_succeed = data_arx['acid_gs_adv_x'], data_arx['acid_gs_succeed']
    yt = data_arx['img_yt']
    mask_config = get_default_mask_config()
    mask_config.update(dict(l1_lambda=1e-4, tv_lambda=1e-2, noise_std=0, n_iters=300,
                       batch_size=50, verbose=False))
    model = resnet50(True)
    model.train(False)
    freeze_model(model)
    if cuda:
        model.to('cuda')
    robj = dict(gs_trans_mask=generate_masks_v2(mask_config,
                                (model, imagenet_normalize, (224, 224)),
                                (acid_gs_adv_x, yt), cuda))
    del model
    return robj


def test_rts(data_arx, cuda):
    acid_gs_adv_x, acid_gs_succeed = data_arx['acid_gs_adv_x'], data_arx['acid_gs_succeed']
    yt = data_arx['img_yt']
    rts_config = get_default_rts_config('resnet50')
    rts_config['batch_size'] = 30
    rts_model = RTSResnet50(rts_config['ckpt_dir'], cuda)
    freeze_model(rts_model.blackbox_model)
    with torch.no_grad():
        robj = dict(gs_trans_rts=generate_rts(rts_config, rts_model, (acid_gs_adv_x, yt), cuda))
    del rts_model
    return robj


def test_cam(data_arx, cuda):
    acid_gs_adv_x, acid_gs_succeed = data_arx['acid_gs_adv_x'], data_arx['acid_gs_succeed']
    yt = data_arx['img_yt']
    cam_config = get_default_cam_config()
    model_tup, forward_fn = cam_resnet50()
    model_tup[0].train(False)
    if cuda:
        model_tup[0].cuda()
    robj = dict(gs_trans_cam=
                generate_cams(cam_config, model_tup, forward_fn, (acid_gs_adv_x, yt), cuda)[1])
    del model_tup
    return robj


def test(config):
    data_arx = np.load(config.input_path)
    cuda = config.device == 'cuda'
    dobj = {}
    dobj.update(test_rts(data_arx, cuda))
    dobj.update(test_mask(data_arx, cuda))
    dobj.update(test_cam(data_arx, cuda))
    np.savez(config.save_path, **dobj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('save_path')
    parser.add_argument('--device', choices=['cpu', 'cuda'], dest='device')

    config = parser.parse_args()
    if config.device is None:
        if torch.cuda.is_available():
            config.device = 'cuda'
        else:
            config.device = 'cpu'
    print('Please check the configuration', config)
    test(config)
