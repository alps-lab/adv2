#!/usr/bin/env python
import numpy as np
import torch

from expr_attacks.cam_model import CAM
from expr_attacks.cam_model_def import cam_resnet50


def get_default_cam_config():
    return dict(batch_size=40)


def generate_cam_per_batch(cam_config, cam_model, model_tup, forward_fn, batch_tup, cuda):
    bx, by = batch_tup
    bx, by = torch.tensor(bx), torch.tensor(by)
    if cuda:
        bx, by = bx.cuda(), by.cuda()
    return cam_model(model_tup, forward_fn, bx, by)[1]


def generate_cams(cam_config, model_tup, forward_fn, images_tup, cuda):
    if cam_config is None:
        cam_config = get_default_cam_config()
    cam_model = CAM()
    img_x, img_y = images_tup[:2]
    batch_size = cam_config['batch_size']
    num_batches = (len(img_x) + batch_size - 1) // batch_size

    cams = []
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min(len(img_x), start_index + batch_size)
        bx, by = img_x[start_index:end_index], img_y[start_index:end_index]
        cams.append(generate_cam_per_batch(cam_config, cam_model, model_tup, forward_fn, (bx, by), cuda).detach().cpu().numpy())

    cams = np.concatenate(cams, axis=0)
    cams_n = cams - np.min(cams, axis=(1, 2, 3), keepdims=True)
    cams_n = cams_n / np.max(cams_n, axis=(1, 2, 3), keepdims=True)

    return cams, cams_n
