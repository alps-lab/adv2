from pathlib import Path
import os
import csv

import numpy as np
import cv2


def load_images_list(path):
    base_dir = str(Path(path).parent)
    paths = []
    labels = []
    with open(path) as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
        for row in rows[1:]:
            paths.append(os.path.join(base_dir, row[0]))
            labels.append(int(row[1]))
    return paths, np.asarray(labels, dtype=np.int64)


def load_images(paths, shape=(224, 224)):
    imgs = []
    for path in paths:
        img = cv2.imread(path)
        img = cv2.resize(img, shape, interpolation=cv2.INTER_LINEAR)
        img = img[..., ::-1].transpose([2, 0, 1]).copy()
        img = np.float32(img) / 255.
        imgs.append(img)

    return np.stack(imgs)


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad_(False)

