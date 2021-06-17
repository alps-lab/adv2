#!/usr/bin/env python
import os
import csv
import cv2
import argparse

import numpy as np

DATA_DIR = '/home/xinyang/Data/intattack/imagenet_samples/detector_val'
CSV_PATH = 'images_d.csv'


def main(config):
    rs = np.random.RandomState(config.seed)
    with open(os.path.join(DATA_DIR, CSV_PATH)) as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
    rows = rows[1:]
    indices = rs.choice(len(rows), 1000, replace=False)

    benign_imgs = []
    benign_ys = []
    benign_paths = []

    for index in indices:
        path = rows[index][0]
        img = cv2.imread(os.path.join(DATA_DIR, path))
        img = cv2.resize(img, (224, 224))
        img = img.transpose([2, 0, 1])[::-1]
        img = np.float32(img / 255.)

        y = int(rows[index][1])
        benign_imgs.append(img)
        benign_ys.append(y)
        benign_paths.append(path)

    benign_imgs = np.stack(benign_imgs)
    benign_ys = np.stack(benign_ys).astype(np.int64)
    benign_paths = np.stack(benign_paths)

    d = {
        'indices': indices,
        'train_benign_x': benign_imgs[:500],
        'train_benign_y': benign_ys[:500],
        'train_benign_path': benign_paths[:500],
        'val_benign_x': benign_imgs[500:],
        'val_benign_y': benign_ys[500],
        'val_benign_path': benign_paths[500:]
    }
    np.savez(config.save_path, **d)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('save_path')
    parser.add_argument('-s', '--seed', dest='seed', type=int)
    main(parser.parse_args())
