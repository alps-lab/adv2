#!/usr/bin/env python
import argparse

import numpy as np
import torch
import torch.nn.functional as F

l1_norm = 56 * 56
l2_norm = 56


def mean_std(x, axis=None):
    return np.mean(x, axis=axis), np.std(x, axis=axis, ddof=1)


def main(config):
    benign_rts_dobj = np.load(config.benign_rts_path)
    adv_rts_dobj = np.load(config.adv_rts_path)
    target_step = 1200

    adv_rts_key = 'pgd_step_%d_adv_rts_yt' % target_step
    benign_rts_key = 'saliency_benign_y'
    benign_rts = benign_rts_dobj[benign_rts_key]
    adv_rts = adv_rts_dobj[adv_rts_key]
    diff = benign_rts - adv_rts
    diff = diff.reshape((len(diff), -1))
    l1_dist = np.linalg.norm(diff, 1, axis=-1) / l1_norm
    l2_dist = np.linalg.norm(diff, 2, axis=-1) / l2_norm
    l1_dist_mu, l1_dist_std = mean_std(l1_dist, -1)
    l2_dist_mu, l2_dist_std = mean_std(l2_dist, -1)

    print('remark', config.remark, 'l1 distance', l1_dist_mu, '±',
          l1_dist_std, 'l2 distance', l2_dist_mu, '±', l2_dist_std)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('benign_rts_path')
    parser.add_argument('adv_rts_path')
    parser.add_argument('-r' '--remark', dest='remark')
    main(parser.parse_args())
