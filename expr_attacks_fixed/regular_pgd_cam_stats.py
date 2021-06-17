#!/usr/bin/env python
import argparse

import numpy as np
import torch
import torch.nn.functional as F

l1_norm = 7 * 7
l2_norm = 7


def mean_std(x, axis=None):
    return np.mean(x, axis=axis), np.std(x, axis=axis, ddof=1)


def main(config):
    benign_cam_dobj = np.load(config.benign_cam_path)
    adv_cam_dobj = np.load(config.adv_cam_path)
    target_step = 1200

    adv_cam_key = 'pgd_step_%d_adv_cam_yt_n' % target_step
    benign_cam_key = 'cam_benign_y_n'
    benign_cams = benign_cam_dobj[benign_cam_key]
    adv_cams = adv_cam_dobj[adv_cam_key]
    diff = benign_cams - adv_cams
    diff = diff.reshape((len(diff), -1))
    l1_dist = np.linalg.norm(diff, 1, axis=-1) / l1_norm
    l2_dist = np.linalg.norm(diff, 2, axis=-1) / l2_norm
    l1_dist_mu, l1_dist_std = mean_std(l1_dist, -1)
    l2_dist_mu, l2_dist_std = mean_std(l2_dist, -1)

    print('remark', config.remark, 'l1 distance', l1_dist_mu, '±',
          l1_dist_std, 'l2 distance', l2_dist_mu, '±', l2_dist_std)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('benign_cam_path')
    parser.add_argument('adv_cam_path')
    parser.add_argument('-r' '--remark', dest='remark')
    main(parser.parse_args())
