#!/usr/bin/env python
import argparse
import os
from collections import OrderedDict

import numpy as np


BENIGN_DIR = '/home/ningfei/data/data_resnet'
BENIGN_CAM_DIR = '/home/ningfei/data/resnet_cam/'
BENIGN_MASK_DIR = '/home/ningfei/data/resnet_mask'
BENIGN_RTS_DIR = '/home/ningfei/data/resnet_benign_rts/'

BENIGN_FMT = 'fold_%d.npz'
BENIGN_CAM_FMT = 'cam_attack_resnet50_cam_benign_fold_resnet_%d.npz'
BENIGN_MASK_FMT = 'mask_attack_resnet50_mask_benign_fold_resnet_%d.npz'
BENIGN_RTS_FMT = 'resnet50_rts_benign_fold_resnet_%d.npz'

ACID_CAM_DIR = '/home/ningfei/data/cam_attack_resnet'
ACID_MASK_DIR = '/home/ningfei/data/resnet_acid_mask_pgd'
ACID_RTS_DIR = '/home/ningfei/data/resnet50_rts_acid_re'

ACID_CAM_FMT = 'resnet50_cam_acid_fold_resnet_%d.npz'
ACID_MASK_FMT = 'resnet50_mask_acid_fold_%d.npz'
ACID_RTS_FMT = 'resnet_rts_acid_pgd_fold_resnet_%d_test.npz'

FOLDS = [1, 2, 3]


def extract_and_concat(paths, keys):
    out_dicts = OrderedDict()
    for key in keys:
        out_dicts[key] = []

    for path in paths:
        data_arx = np.load(path)
        for key in keys:
            out_dicts[key].append(data_arx[key])
    out_dicts = {key: np.concatenate(arr, axis=0) for key, arr in out_dicts.items()}
    return list(out_dicts.values())


def extract_cam_acid():
    adv_step_template = 'pgd_s2_step_%d_adv_x'
    cam_step_template = 'pgd_s2_step_%d_adv_cam'
    cam_n_step_template = 'pgd_s2_step_%d_adv_cam_n'
    logits_step_template = 'pgd_s2_step_%d_adv_logits'

    return extract_and_concat([os.path.join(ACID_CAM_DIR, ACID_CAM_FMT % fold) for fold in
                               FOLDS], ['pgd_s2_step_%d_adv_x' % 1400, 'pgd_s2_step_%d_adv_succeed' % 1400,
                                        'pgd_s2_step_%d_adv_cam_n' % 1400])


def extract_mask_acid(benign_mask):
    adv_disc_step_template = 'pgd_s2_step_%d_adv_x_disc'
    mask_disc_mnow_step_template = 'pgd_s2_step_%d_adv_mask_mnow_disc'  # current optimal MASK
    succeed_disc_step_tamplate = 'pgd_s2_step_%d_succeed_disc'

    n = len(benign_mask)
    best_l2_dists = np.full((n,), np.inf, np.float32)
    best_adv_x = np.zeros((n, 3, 224, 224), np.float32)
    best_mask = np.zeros((n, 1, 28, 28), np.float32)
    for step in range(49, 1200, 50):
        step = step + 1
        adv_x, succeed, mask = extract_and_concat([os.path.join(ACID_MASK_DIR, ACID_MASK_FMT % fold) for fold in
                               FOLDS], ['pgd_s2_step_%d_adv_x_disc' % step, 'pgd_s2_step_%d_succeed_disc' % step,
                                        'pgd_s2_step_%d_adv_mask_mnow_disc' % step])
        diff = mask.reshape((n, -1)) - benign_mask.reshape((n, -1))
        l2_dists = np.linalg.norm(diff, 2, axis=1)
        update_flag = l2_dists < best_l2_dists
        best_adv_x[update_flag] = adv_x[update_flag]
        best_mask[update_flag] = mask[update_flag]
        best_l2_dists[update_flag] = l2_dists[update_flag]

    succeed = (best_l2_dists < np.inf).astype(np.int64)
    return best_adv_x, succeed, best_mask


def extract_rts_acid():
    adv_step_template = 'pgd_s2_step_%d_adv_x'
    cam_step_template = 'pgd_s2_step_%d_adv_cam'
    cam_n_step_template = 'pgd_s2_step_%d_adv_cam_n'
    logits_step_template = 'pgd_s2_step_%d_adv_logits'

    return extract_and_concat([os.path.join(ACID_RTS_DIR, ACID_RTS_FMT % fold) for fold in
                               FOLDS], ['pgd_s2_step_%d_adv_x' % 1400, 'pgd_s2_step_%d_adv_succeed' % 1400,
                                        'pgd_s2_step_%d_adv_rts' % 1400])


def main(config):
    img_x, img_y, img_yt = extract_and_concat([os.path.join(BENIGN_DIR, BENIGN_FMT % fold) for fold in FOLDS],
                                              ['img_x', 'img_y', 'img_yt'])
    benign_cam, = extract_and_concat([os.path.join(BENIGN_CAM_DIR, BENIGN_CAM_FMT % fold) for fold in FOLDS],
                                        ['cam_benign_y_n'])
    benign_mask, = extract_and_concat([os.path.join(BENIGN_MASK_DIR, BENIGN_MASK_FMT %fold) for fold in FOLDS],
                                     ['mask_benign_y'])
    benign_rts, = extract_and_concat([os.path.join(BENIGN_RTS_DIR, BENIGN_RTS_FMT % fold) for fold in FOLDS],
                                    ['saliency_benign_y'])

    acid_cam_adv_x, acid_cam_succeed, acid_cam_cam = extract_cam_acid()

    acid_mask_adv_x, acid_mask_succeed, acid_mask_mask = extract_mask_acid(benign_mask)

    acid_rts_adv_x, acid_rts_succeed, acid_rts_rts = extract_rts_acid()

    save_dict = locals()
    del save_dict['config']
    np.savez(config.save_path, **save_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('save_path')
    parser.add_argument('--fold-size', type=int, dest='fold_size')

    config = parser.parse_args()
    print('Please check the configuration', config)
    main(config)
