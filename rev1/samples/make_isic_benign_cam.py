#!/usr/bin/env python
import argparse

import torch
import numpy as np

from expr_attacks.cam_generator import generate_cam_per_batch
from expr_attacks.cam_model import CAM
from rev1.isic_acid.isic_cam_model_def import cam_resnet50


def main(config):
    if config.device is None:
        config.device = 'gpu' if torch.cuda.is_available() else 'cpu'

    data_arx = np.load(config.data_path)
    img_x, img_y, img_yt = data_arx['img_x'], data_arx['img_y'], data_arx['img_yt']

    model_tup, (cam_resnet50_forward, cam_resnet50_fc_weight) = cam_resnet50()
    cam_model = CAM()
    if config.device == 'gpu':
        model_tup[0].cuda()

    n = len(img_x)
    batch_size = config.batch_size
    num_batches = (n + config.batch_size - 1) // batch_size
    cams = []
    for i in range(num_batches):
        si = i * batch_size
        ei = min(n, si + batch_size)
        bx, by = img_x[si:ei], img_y[si:ei]
        cam = generate_cam_per_batch(None, cam_model, model_tup, (cam_resnet50_forward, cam_resnet50_fc_weight),
                                            (bx, by), config.device == 'gpu')
        cams.append(cam)

    cams = np.concatenate(cams)
    cam_ns = cams - cams.min(axis=(1, 2, 3), keepdims=True)
    cam_ns = cam_ns / cam_ns.max(axis=(1, 2, 3,), keepdims=True)
    dobj = dict(data_arx.items())
    dobj['cam_benign_y'] = cams
    dobj['cam_benign_y_n'] = cam_ns
    np.savez(config.save_path, **dobj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('save_path')
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=64)
    parser.add_argument('-d', '--device', dest='device', choices=['cpu', 'gpu'])

    main(parser.parse_args())
