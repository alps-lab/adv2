#!/usr/bin/env python
import argparse

import numpy as np
import torch
import torch.nn as nn

from rev2.cifar10.model_utils import resnet50, CIFAR10_RESNET50_CKPT_PATH
from rev2.gs.generate_gs import generate_gs
from rev2.cifar10.data_utils import cifar10_normalize
from rev2.cifar10.generate_gs_benign import cifar10_resize_postfn


def load_model(config):
    model = resnet50()
    nn.DataParallel(model).load_state_dict(
        torch.load(CIFAR10_RESNET50_CKPT_PATH, lambda storage, location: storage)['net']
    )
    model.to(config.device)
    model.train(False)
    return model, cifar10_normalize


def main(config):
    model_tup = load_model(config)

    dobj = np.load(config.data_path)
    adv_dobj = np.load(config.adv_data_path)
    img_x, img_yt = adv_dobj['pgd_step_1500_adv_x'], dobj['img_yt']
    pgd_gs = generate_gs(model_tup, img_x, img_yt, cifar10_resize_postfn, False, batch_size=50)
    save_dobj = {'pgd_x': img_x, 'img_yt': img_yt, 'pgd_gs': pgd_gs}
    np.savez(config.save_path, **save_dobj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('adv_data_path')
    parser.add_argument('save_path')
    parser.add_argument('-d', '--device', dest='device', choices=['cpu', 'cuda'])
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=50)
    config = parser.parse_args()

    if config.device is None:
        if torch.cuda.is_available():
            config.device = 'cuda'
        else:
            config.device = 'cpu'
    print('configuration:', config)
    main(config)
