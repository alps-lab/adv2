#!/usr/bin/env python
import argparse

import torch
import numpy as np

from expr_attacks.mask_generator import generate_mask_per_batch_v2, get_default_mask_config
from expr_attacks.mask_model import MASKV2
from rev1.isic_acid.isic_model import get_isic_model_on_resnet50
from rev1.isic_acid.isic_utils import ISIC_RESNET50_CKPT_PATH


def identity(x):
    return x


def main(config):
    if config.device is None:
        config.device = 'gpu' if torch.cuda.is_available() else 'cpu'

    data_arx = np.load(config.adv_data_path)
    adv_x, img_yt = data_arx['adv_x'], data_arx['img_yt']

    model = get_isic_model_on_resnet50(ckpt_path=ISIC_RESNET50_CKPT_PATH)
    model_tup = (model, identity, (224, 224))

    mask_model = MASKV2(config.device == 'gpu')
    if config.device == 'gpu':
        model_tup[0].cuda()

    n = len(adv_x)
    batch_size = config.batch_size
    num_batches = (n + config.batch_size - 1) // batch_size
    masks = []

    mask_config = get_default_mask_config()
    mask_config.update(dict(lr=0.1, l1_lambda=1e-4, tv_lambda=1e-2, noise_std=0, n_iters=300,
                       batch_size=40, verbose=False))
    for i in range(num_batches):
        si = i * batch_size
        ei = min(n, si + batch_size)
        bx, by = adv_x[si:ei], img_yt[si:ei]
        mask = generate_mask_per_batch_v2(mask_config, mask_model, model_tup,
                                            (bx, by), config.device == 'gpu')
        masks.append(mask.detach().cpu().numpy())

    masks = np.concatenate(masks)
    dobj = dict(data_arx.items())
    dobj['mask_pgd_yt'] = masks
    np.savez(config.save_path, **dobj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('adv_data_path')
    parser.add_argument('save_path')
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=64)
    parser.add_argument('-d', '--device', dest='device', choices=['cpu', 'gpu'])

    main(parser.parse_args())
