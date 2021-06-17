#!/usr/bin/env python
import argparse
import os

import cv2
import numpy as np

from expr_shape.utils import generate_shapes_v2
from scipy.spatial.distance import cdist


IN_DIR = '/home/xinyang/Data/intattack/fixup1/resnet_data/'
RTS_IN_DIR = '/home/xinyang/Data/intattack/rev1/benign_maps/mask_v2_resnet50_benign/'
OUT_DIR = '/home/xinyang/Data/intattack/rev1/random_shape/tmask'
FOLD_SIZE = 30


if __name__ == '__main__':
    imgs, ys, yts, bmasks, tmasks = [], [], [], [], []
    for i in [7]:
        fold_num = i
        data_arx = np.load(os.path.join(IN_DIR, 'fold_%d.npz' % fold_num))
        bmap_arx = np.load(os.path.join(RTS_IN_DIR, 'fold_%d.npz' % fold_num))
        imgs.append(data_arx['img_x'])
        ys.append(data_arx['img_y'])
        yts.append(data_arx['img_yt'])
        bmasks.append(bmap_arx['mask_benign_y'])
    imgs = np.concatenate(imgs, axis=0)
    ys = np.concatenate(ys, axis=0)
    yts = np.concatenate(yts, axis=0)
    bmasks = np.concatenate(bmasks, axis=0)

    n1, n2 = len(imgs), 100
    target_maps = generate_shapes_v2(n2)
    resized = [cv2.resize(target_map, (28, 28))[None] for target_map in target_maps]
    tmasks = np.stack(resized).astype(np.float32)
    tmasks = 1 - tmasks

    att_imgs = []
    att_ys = []
    att_yts = []
    att_bmasks = []
    att_tmasks = []
    counter = {i: 0 for i in range(n1)}
    distmat = np.argsort(cdist(tmasks.reshape((n2, -1)), bmasks.reshape((n1, -1))), axis=1)[:, ::-1]
    for i in range(n2):
        for j in range(n1):
            idx = int(distmat[i][j])
            if counter[idx] < 3:
                counter[idx] += 1
                att_imgs.append(imgs[idx])
                att_ys.append(ys[idx])
                att_yts.append(yts[idx])
                att_bmasks.append(bmasks[idx])
                att_tmasks.append(tmasks[i])
                break
    att_imgs = np.stack(att_imgs)
    att_ys = np.stack(att_ys)
    att_yts = np.stack(att_yts)
    att_bmasks = np.stack(att_bmasks)
    att_tmasks = np.stack(att_tmasks)

    n_folds = (n2 + FOLD_SIZE - 1) // FOLD_SIZE
    os.makedirs(OUT_DIR, exist_ok=True)
    for i in range(n_folds):
        si = i * FOLD_SIZE
        ei = min(n2, si + FOLD_SIZE)
        np.savez(os.path.join(OUT_DIR, 'fold_%d.npz' % (i + 1)),
                 att_imgs=att_imgs[si:ei], att_ys=att_ys[si:ei],
                 att_yts=att_yts[si:ei], att_bmasks=att_bmasks[si:ei],
                 att_tmasks=att_tmasks[si:ei])
