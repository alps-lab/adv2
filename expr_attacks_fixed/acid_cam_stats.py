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

    adv_cam_key = 'pgd_s2_step_%d_adv_cam_n' % target_step
    benign_cam_key = 'cam_benign_y_n'
    adv_logits_key = 'pgd_s2_step_%d_adv_logits' % target_step
    adv_succeed_key = 'pgd_s2_step_%d_adv_succeed' % target_step
    benign_cams = benign_cam_dobj[benign_cam_key]
    adv_cams = adv_cam_dobj[adv_cam_key]
    adv_logits = adv_cam_dobj[adv_logits_key]
    adv_succeed = adv_cam_dobj[adv_succeed_key]

    succeed = adv_succeed.astype(np.bool)
    adv_probs = F.softmax(torch.from_numpy(adv_logits), dim=-1).max(1)[0].numpy()
    adv_probs = adv_probs[succeed]
    adv_probs_mu, adv_probs_std = mean_std(adv_probs, -1)

    diff = benign_cams - adv_cams
    diff = diff.reshape((len(diff), -1))[succeed]
    l1_dist = np.linalg.norm(diff, 1, axis=-1) / l1_norm
    l2_dist = np.linalg.norm(diff, 2, axis=-1) / l2_norm
    l1_dist_mu, l1_dist_std = np.mean(l1_dist), np.std(l1_dist, ddof=1)
    l2_dist_mu, l2_dist_std = np.mean(l2_dist), np.std(l2_dist, ddof=1)

    print('remark', config.remark, 'succeed', np.mean(succeed.astype(np.int64)),
          'confidence', adv_probs_mu, '±', adv_probs_std,
          'l1 distance', l1_dist_mu, '±',
          l1_dist_std, 'l2 distance', l2_dist_mu, '±', l2_dist_std)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('benign_cam_path')
    parser.add_argument('adv_cam_path')
    parser.add_argument('-r' '--remark', dest='remark')
    main(parser.parse_args())
