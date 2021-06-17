#!/usr/bin/env python
import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.models import densenet169
from ia_utils.data_utils import imagenet_normalize


RESNET50_BENIGN_IMGS_DIR = '/home/ningfei/data/data_resnet/'
RESNET50_ADV_IMGS_DIR = '/home/ningfei/data/data_resnet_2/'
RESNET50_BENIGN_NAME_FMT = 'fold_%d.npz'
RESNET50_ADV_NAME_FMT = 'mask_attack_pgd_vanilla_fold_resnet_%d.npz'
RESNET50_FOLDS = [1, 2, 3]
RESNET50_BENIGN_IMGS_PATHS = [os.path.join(RESNET50_BENIGN_IMGS_DIR, RESNET50_BENIGN_NAME_FMT % fold) for fold in
                              RESNET50_FOLDS]
RESNET50_ADV_IMGS_PATHS = [os.path.join(RESNET50_ADV_IMGS_DIR, RESNET50_ADV_NAME_FMT % fold) for fold
                           in RESNET50_FOLDS]

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
CUDA = True


def read_images(which):
    if which == 'resnet50':
        benign_paths, adv_paths = RESNET50_BENIGN_IMGS_PATHS, RESNET50_ADV_IMGS_PATHS
    if which == 'densenet169':
        benign_paths, adv_paths = DENSENET169_BENIGN_IMGS_PATHS, DENSENET169_ADV_IMGS_PATH

    img_x, img_y, img_path, img_yt, adv_x, adv_succeed = [[] for _ in range(6)]
    for res_benign_path, res_adv_path in zip(benign_paths, adv_paths):
        print(res_benign_path, res_adv_path)
        benign_arx, adv_arx = np.load(res_benign_path), np.load(res_adv_path)
        img_x.append(benign_arx['img_x'])
        img_y.append(benign_arx['img_y'])
        img_yt.append(benign_arx['img_yt'])
        img_path.append(benign_arx['img_path'])
        adv_x.append(adv_arx['pgd_step_%d_adv_x' % ADV_STEP])
        adv_succeed.append(adv_arx['pgd_step_%d_succeed' % ADV_STEP])
    img_x = np.concatenate(img_x, axis=0)
    img_y = np.concatenate(img_y, axis=0)
    img_yt = np.concatenate(img_yt, axis=0)
    adv_x = np.concatenate(adv_x, axis=0)
    adv_succeed = np.concatenate(adv_succeed, axis=0)
    img_path = np.concatenate(img_path, axis=0)
    index = np.arange(len(img_x), dtype=np.int64)

    df = pd.DataFrame({'index': index, 'adv_succeed': adv_succeed, 'path': img_path})
    df = df.set_index('path')
    return (img_x, img_y, img_yt, adv_x), df


def test_labels(model, pre_fn, img_x, desired_y):
    n, batch_size = len(img_x), 40
    n_batches = (n + batch_size - 1) // batch_size
    preds = []

    for i in range(n_batches):
        si = i * batch_size
        ei = min(n, si + batch_size)
        bx_np = img_x[si:ei]
        bx = torch.tensor(bx_np)
        if CUDA:
            bx = bx.cuda()
        with torch.no_grad():
            preds.append(model(pre_fn(bx)).argmax(1).cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    return (preds != desired_y).astype(np.int64)


def summary_informations():
    resnet_data, resnet_df = read_images('resnet50')
    densenet_data, densenet_df = read_images('densenet169')

    df_all = pd.merge(resnet_df, densenet_df, left_index=True, right_index=True, suffixes=['_resnet', '_densenet'])
    int_resnet_index = df_all[df_all['adv_succeed_resnet'] == 1]['index_resnet'].values
    int_densenet_index = df_all[df_all['adv_succeed_densenet'] == 1]['index_densenet'].values
    int_resnet_adv = resnet_data[3][int_resnet_index]
    int_resnet_y = resnet_data[1][int_resnet_index]
    int_resnet_yt = resnet_data[2][int_resnet_index]
    int_densenet_adv = densenet_data[3][int_densenet_index]
    int_densenet_y = densenet_data[1][int_densenet_index]
    int_densenet_yt = densenet_data[2][int_densenet_index]

    resnet_model = resnet50(True)
    densenet_model = densenet169(True)
    if CUDA:
        resnet_model.cuda()
        densenet_model.cuda()
    resnet_model.train(False)
    densenet_model.train(False)
    resnet_flag = test_labels(resnet_model, imagenet_normalize, int_densenet_adv, int_densenet_y)
    densenet_flag = test_labels(densenet_model, imagenet_normalize, int_resnet_adv, int_resnet_y)

    print('resnet_flag', len(resnet_flag), np.count_nonzero(resnet_flag))
    print('densenet_flag', len(densenet_flag), np.count_nonzero(densenet_flag))


if __name__ == '__main__':
    summary_informations()
