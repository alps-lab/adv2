#!/usr/bin/env python
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


RESNET50_BENIGN_IMGS_DIR = '/home/ningfei/data/data_resnet/'
RESNET50_ADV_IMGS_DIR = '/home/ningfei/data/data_resnet_2/'
RESNET50_BENIGN_NAME_FMT = 'fold_%d.npz'
RESNET50_ADV_NAME_FMT = 'mask_attack_pgd_vanilla_fold_resnet_%d.npz'
RESNET50_FOLDS = [1, 2, 3]
RESNET50_BENIGN_IMGS_PATHS = [os.path.join(RESNET50_BENIGN_IMGS_DIR, RESNET50_BENIGN_NAME_FMT % fold) for fold in
                              RESNET50_FOLDS]
RESNET50_ADV_IMGS_PATHS = [os.path.join(RESNET50_ADV_IMGS_DIR)]


DENSENET169_BENIGN_IMGS_DIR = '/home/ningfei/data/densenet_data/'
DENSENET169_ADV_IMGS_DIR = '/home/ningfei/data/densenet_adv_sam/'
DENSENET169_BENIGN_NAME_FMT = 'fold_%d.npz'
DENSENET169_ADV_NAME_FNT = 'densenet169_pgd_vanilla_fold_%d.npz'
DENSENET169_FOLDS = [1, 2, 3]
DENSENET169_BENIGN_IMGS_PATHS = [os.path.join(DENSENET169_BENIGN_IMGS_DIR, DENSENET169_BENIGN_NAME_FMT % fold)
                                 for fold in DENSENET169_FOLDS]
DENSENET169_ADV_IMGS_PATH = [os.path.join(DENSENET169_ADV_IMGS_DIR, DENSENET169_ADV_NAME_FNT % fold)
                             for fold in DENSENET169_FOLDS]


ADV_STEP = 1000


def read_images():
    img_x, img_y, img_yt, adv_x, adv_succeed = [[] for _ in range(5)]
    for res_benign_path, res_adv_path in zip(RESNET50_BENIGN_IMGS_PATHS, RESNET50_ADV_IMGS_PATHS):
        benign_arx, adv_arx = np.load(res_benign_path), np.load(res_adv_path)
        img_x.append(benign_arx['img_x'])
        img_y.append(benign_arx['img_y'])
        img_yt.append(benign_arx['img_yt'])
        adv_x.append(adv_arx['pgd_step_%d_adv_x' % ADV_STEP])
        adv_succeed.append(adv_arx['pgd_step_%d_succeed' % ADV_STEP])

def summary_informations():
    pass


if __name__ == '__main__':
    pass
