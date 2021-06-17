#!/usr/bin/env python
import argparse
import os

import cv2
import numpy as np

from expr_shape.utils import generate_shapes
from scipy.spatial.distance import cdist


IN_DIR = '/home/xinyang/Data/intattack/random_shape/data'
RTS_IN_DIR = '/home/xinyang/Data/intattack/random_shape/brts'
OUT_DIR = '/home/xinyang/Data/intattack/random_shape/trts'
FOLD_SIZE = 100


if __name__ == '__main__':
    imgs, ys, yts, brts, trts = [], [], [], [], []
    for i in range(3):
        fold_num = i + 1
        data_arx = np.load(os.path.join(IN_DIR, 'fold_%d.npz' % fold_num))
        bmap_arx = np.load(os.path.join(RTS_IN_DIR, 'fold_%d.npz' % fold_num))
        imgs.append(data_arx['img_x'])
        ys.append(data_arx['img_y'])
        yts.append(data_arx['img_yt'])
        brts.append(bmap_arx['saliency_benign_y'])
    imgs = np.concatenate(imgs, axis=0)
    ys = np.concatenate(ys, axis=0)
    yts = np.concatenate(yts, axis=0)
    brts = np.concatenate(brts, axis=0)

    n1, n2 = len(imgs), 100
    target_maps = generate_shapes(n2)
    resized = [cv2.resize(target_map, (56, 56))[None] for target_map in target_maps]
    trts = np.stack(resized).astype(np.float32)

    att_imgs = []
    att_ys = []
    att_yts = []
    att_brts = []
    att_trts = []
    counter = {i: 0 for i in range(n1)}
    distmat = np.argsort(cdist(trts.reshape((n2, -1)), brts.reshape((n1, -1))), axis=1)[:, ::-1]
    for i in range(n2):
        for j in range(n1):
            idx = int(distmat[i][j])
            if counter[idx] < 3:
                counter[idx] += 1
                att_imgs.append(imgs[idx])
                att_ys.append(ys[idx])
                att_yts.append(yts[idx])
                att_brts.append(brts[idx])
                att_trts.append(trts[i])
                break
    att_imgs = np.stack(att_imgs)
    att_ys = np.stack(att_ys)
    att_yts = np.stack(att_yts)
    att_brts = np.stack(att_brts)
    att_trts = np.stack(att_trts)

    n_folds = (n2 + FOLD_SIZE - 1) // FOLD_SIZE
    os.makedirs(OUT_DIR, exist_ok=True)
    for i in range(n_folds):
        si = i * FOLD_SIZE
        ei = min(n2, si + FOLD_SIZE)
        np.savez(os.path.join(OUT_DIR, 'fold_%d.npz' % (i + 1)), att_imgs=att_imgs, att_ys=att_ys, att_yts=att_yts, att_brts=att_brts, att_trts=att_trts)
