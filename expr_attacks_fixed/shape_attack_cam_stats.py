#!/usr/bin/env python
import sys

import torch
import torch.nn.functional as F
import numpy as np


def mean_std(x):
    mu = np.mean(x)
    std = np.std(x)

    return "%f ± %f" % (mu, std)


def compute_confidence(logits):
    probs = F.softmax(torch.from_numpy(logits), dim=-1).numpy()
    return probs.max(axis=1).mean()


if __name__ == '__main__':
    assert len(sys.argv) == 2, 'bad path'
    l1_norm = 7 * 7
    l2_norm = 7

    h = np.load(sys.argv[1])
    bcam = h['bcam']
    tcam = h['tcam']
    acam = h['adv_cam']
    n = len(bcam)
    bcam, tcam, acam = bcam.reshape((n, -1)), tcam.reshape((n, -1)), acam.reshape((n, -1))
    diff_b = bcam - tcam
    diff_a = acam - tcam

    dist_l1_b = np.linalg.norm(diff_b, 1, axis=-1) / l1_norm
    dist_l2_b = np.linalg.norm(diff_b, 2, axis=-1) / l2_norm
    dist_l1_a = np.linalg.norm(diff_a, 1, axis=-1) / l1_norm
    dist_l2_a = np.linalg.norm(diff_a, 2, axis=-1) / l2_norm

    print("l1 dist(bcam, tcam)", mean_std(dist_l1_b))
    print("l2 dist(bcam, tcam", mean_std(dist_l2_b))
    print("l1 dist(adv_cam, tcam)", mean_std(dist_l1_a))
    print("l2 dist(adv_cam, bcam)", mean_std(dist_l2_a))
    print('succeed', h['adv_succeed'].mean())
    print('acid confidence', compute_confidence(h['adv_logits']))