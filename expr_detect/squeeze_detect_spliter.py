import os
import argparse

import numpy as np

from expr_detect.commons import NUM_FOLDS


def main(config):
    img_xs, img_ys, img_yts = [], [], []
    for i in range(NUM_FOLDS):
        fold = i + 1
        img_arx = np.load(os.path.join(config.image_dir, 'fold_%d.npz' % fold))
        img_x, img_y, img_yt = img_arx['img_x'], img_arx['img_y'], img_arx['img_yt']
        img_xs.append(img_x)
        img_ys.append(img_y)
        img_yts.append(img_yt)
    img_xs = np.concatenate(img_xs, 0)
    img_ys = np.concatenate(img_ys, 0)
    img_yts = np.concatenate(img_yts, 0)
    print(img_xs.shape, img_ys.shape, img_yts.shape)

    n = len(img_xs)
    indices = np.random.RandomState(config.seed).choice(n, size=(500 + 500 + 100 + 100), replace=False)
    train_benign_indices = indices[:500]
    val_benign_indices = indices[500:1000]
    train_adv_indices = indices[1000:110]
    val_adv_indices = indices[1100:1200]

    d = {
        'indices': indices,
        'train_benign_x': img_xs[train_benign_indices],
        'train_benign_y': img_ys[train_benign_indices],
        'val_benign_x': img_xs[val_benign_indices],
        'val_benign_y': img_ys[val_benign_indices],
        'train_adv_x': img_xs[train_adv_indices],
        'train_adv_y': img_ys[train_adv_indices],
        'train_adv_yt': img_yts[train_adv_indices],
        'val_adv_x': img_xs[val_adv_indices],
        'val_adv_y': img_ys[val_adv_indices],
        'val_adv_yt': img_yts[val_adv_indices]
    }
    np.savez(config.save_path, **d)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['resnet50', 'densenet169'])
    parser.add_argument('image_dir')
    parser.add_argument('save_path')
    parser.add_argument('-s', '--seed', type=int, dest='seed')

    main(parser.parse_args())
