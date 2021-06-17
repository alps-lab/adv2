#!/usr/bin/env python
import argparse

import torch
import numpy as np

from rev2.gs.generate_gs import generate_gs, imagenet_resize_postfn
from rev1.isic_acid.isic_model import get_isic_model_on_resnet50
from rev1.isic_acid.isic_utils import ISIC_RESNET50_CKPT_PATH


def identity(x):
    return x


def main(config):
    data_arx = np.load(config.adv_data_path)
    adv_x, img_yt = data_arx['adv_x'], data_arx['img_yt']

    model = get_isic_model_on_resnet50(ckpt_path=ISIC_RESNET50_CKPT_PATH)
    model_tup = (model, identity, (224, 224))

    if config.device == 'cuda':
        model_tup[0].cuda()

    n = len(adv_x)
    batch_size = config.batch_size
    num_batches = (n + config.batch_size - 1) // batch_size
    pgd_gss = []

    for i in range(num_batches):
        si = i * batch_size
        ei = min(n, si + batch_size)
        bx, by = adv_x[si:ei], img_yt[si:ei]
        pgd_gs = generate_gs(model_tup, bx, by, imagenet_resize_postfn, False, batch_size=50)
        pgd_gss.append(pgd_gs)

    pgd_gss = np.concatenate(pgd_gss)
    dobj = dict(data_arx.items())
    dobj['pgd_gs'] = pgd_gss
    np.savez(config.save_path, **dobj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('adv_data_path')
    parser.add_argument('save_path')
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=64)
    parser.add_argument('-d', '--device', dest='device', choices=['cpu', 'cuda'])

    config = parser.parse_args()
    if config.device is None:
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('configuration:', config)

    main(config)
