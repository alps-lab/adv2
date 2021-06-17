#!/usr/bin/env python
import argparse

import numpy as np
import torch

from torchvision.models.resnet import resnet50
from rev2.gs.generate_gs import generate_gs, imagenet_resize_postfn
from ia_utils.data_utils import imagenet_normalize


def test_gs(data_arx, cuda):
    device = 'cuda' if cuda else 'cpu'
    acid_rts_adv_x, acid_rts_succeed = data_arx['acid_rts_adv_x'], data_arx['acid_rts_succeed']
    yt = data_arx['img_yt']
    model = resnet50(True)
    model.train(False)
    model.to(device)

    robj = dict(mask_trans_gs=generate_gs((model, imagenet_normalize),
                                         acid_rts_adv_x, yt,
                                         imagenet_resize_postfn,
                                         device=device, batch_size=36),
                img_x=data_arx['img_x'],
                acid_adv_rts=data_arx['acid_rts_rts'],
                acid_adv_x=acid_rts_adv_x)
    return robj



def test(config):
    data_arx = np.load(config.input_path)
    cuda = config.device == 'cuda'
    dobj = {}
    dobj.update(test_gs(data_arx, cuda))
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
