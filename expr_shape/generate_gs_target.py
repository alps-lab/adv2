#!/usr/bin/env python
import argparse
import os

import cv2
import numpy as np

from expr_shape.utils import generate_shapes
from scipy.spatial.distance import cdist


IN_DIR = '/home/xinyang/Data/intattack/rev2/random_shape/data_new'
GS_IN_DIR = '/home/xinyang/Data/intattack/rev2/random_shape/bgs'
OUT_DIR = '/home/xinyang/Data/intattack/rev2/random_shape/tgs'
FOLD_SIZE = 100


if __name__ == '__main__':
    imgs, ys, yts, bgss, tgss = [], [], [], [], []
    for i, b_fold in enumerate([6, 7, 8]):
        data_arx = np.load(os.path.join(IN_DIR, 'fold_%d.npz' % b_fold))
        bmap_arx = np.load(os.path.join(GS_IN_DIR, 'fold_%d.npz' % b_fold))
        imgs.append(data_arx['img_x'])
        ys.append(data_arx['img_y'])
        yts.append(data_arx['img_yt'])
        bgss.append(bmap_arx['benign_gs'])
    imgs = np.concatenate(imgs, axis=0)
    ys = np.concatenate(ys, axis=0)
    yts = np.concatenate(yts, axis=0)
    bgss = np.concatenate(bgss, axis=0)

    n1, n2 = len(imgs), 100
    target_maps = generate_shapes(n2)
    resized = [cv2.resize(target_map, (56, 56))[None] for target_map in target_maps]
    tgss = np.stack(resized).astype(np.float32)

    att_imgs = []
    att_ys = []
    att_yts = []
    att_gss = []
    att_tgss = []
    counter = {i: 0 for i in range(n1)}
    distmat = np.argsort(cdist(tgss.reshape((n2, -1)), bgss.reshape((n1, -1))), axis=1)
    for i in range(n2):
        for j in range(n1):
            idx = int(distmat[i][j])
            if counter[idx] < 3:
                counter[idx] += 1
                att_imgs.append(imgs[idx])
                att_ys.append(ys[idx])
                att_yts.append(yts[idx])
                att_gss.append(bgss[idx])
                att_tgss.append(tgss[i])
                break
    att_imgs = np.stack(att_imgs)
    att_ys = np.stack(att_ys)
    att_yts = np.stack(att_yts)
    att_gss = np.stack(att_gss)
    att_tgss = np.stack(att_tgss)

    n_folds = (n2 + FOLD_SIZE - 1) // FOLD_SIZE
    os.makedirs(OUT_DIR, exist_ok=True)
    for i in range(n_folds):
        si = i * FOLD_SIZE
        ei = min(n2, si + FOLD_SIZE)
        np.savez(os.path.join(OUT_DIR, 'fold_%d.npz' % (i + 1)), att_imgs=att_imgs, att_ys=att_ys, att_yts=att_yts, att_bgs=att_gss, att_tgs=att_tgss)
