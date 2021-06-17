#!/usr/bin/env python
import argparse

from PIL import Image
import numpy as np
import pandas as pd

from rev1.isic_acid.isic_utils import ISIC_EVAL_TRANSFORMS


def main(config):
    rs = np.random.RandomState(None)
    df = pd.read_csv(config.csv_path, header=0)
    paths = df['path'].tolist()
    labels = df['label'].values.copy()
    label_ids, label_counts = np.unique(labels, return_counts=True)
    weights = np.zeros(len(label_ids), dtype=np.float64)
    for label_id, label_count in zip(label_ids, label_counts):
        weights[label_id] = 1. / label_count
    p = weights[labels]
    p = p / p.sum()

    indices = rs.choice(len(paths), size=config.n, replace=False, p=p)
    sampled_paths = [paths[i] for i in indices]
    sampled_labels = labels[indices]

    to_save = []
    for path, label in zip(sampled_paths, sampled_labels):
        image = Image.open(path)
        transformed = ISIC_EVAL_TRANSFORMS(image)
        transformed = transformed.numpy()
        to_save.append(transformed)
    to_save = np.stack(to_save)
    target_labels = np.mod(sampled_labels + rs.randint(1, 7, size=len(sampled_paths)), 7)
    np.savez(config.save_path, img_x=to_save, img_y=sampled_labels, img_yt=target_labels, img_path=sampled_paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path')
    parser.add_argument('-n', type=int, default=100, dest='n')
    parser.add_argument('save_path')

    main(parser.parse_args())
