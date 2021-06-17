#!/usr/bin/env python
import argparse
import os
from collections import OrderedDict

import numpy as np


BENIGN_DIR = '/home/xinyang/Data/intattack/fixup1/resnet_data/'
BENIGN_CAM_DIR = '/home/xinyang/Data/intattack/fixup1/resnet50_cam/'
BENIGN_MASK_DIR = '/home/xinyang/Data/intattack/fixup1/resnet50_mask/'
BENIGN_RTS_DIR = '/home/xinyang/Data/intattack/fixup1/resnet50_rts/'

BENIGN_FMT = 'fold_%d.npz'
BENIGN_CAM_FMT = 'fold_%d.npz'
BENIGN_MASK_FMT = 'fold_%d.npz'
BENIGN_RTS_FMT = 'fold_%d.npz'

ACID_CAM_DIR = '/home/xinyang/Data/intattack/fixup1/resnet50_acid_cam/'
ACID_RTS_DIR = '/home/xinyang/Data/intattack/fixup1/resnet50_acid_rts/'

ACID_CAM_FMT = 'fold_%d.npz'
ACID_RTS_FMT = 'fold_%d.npz'

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
                               FOLDS], ['pgd_s2_step_%d_adv_x' % 1200, 'pgd_s2_step_%d_adv_succeed' % 1200,
                                        'pgd_s2_step_%d_adv_cam_n' % 1200])


def extract_rts_acid():
    adv_step_template = 'pgd_s2_step_%d_adv_x'
    cam_step_template = 'pgd_s2_step_%d_adv_cam'
    cam_n_step_template = 'pgd_s2_step_%d_adv_cam_n'
    logits_step_template = 'pgd_s2_step_%d_adv_logits'

    return extract_and_concat([os.path.join(ACID_RTS_DIR, ACID_RTS_FMT % fold) for fold in
                               FOLDS], ['pgd_s2_step_%d_adv_x' % 1200, 'pgd_s2_step_%d_adv_succeed' % 1200,
                                        'pgd_s2_step_%d_adv_rts' % 1200])


def main(config):
    img_x, img_y, img_yt = extract_and_concat([os.path.join(BENIGN_DIR, BENIGN_FMT % fold) for fold in FOLDS],
                                              ['img_x', 'img_y', 'img_yt'])
    benign_cam, = extract_and_concat([os.path.join(BENIGN_CAM_DIR, BENIGN_CAM_FMT % fold) for fold in FOLDS],
                                        ['cam_benign_y_n'])
    benign_rts, = extract_and_concat([os.path.join(BENIGN_RTS_DIR, BENIGN_RTS_FMT % fold) for fold in FOLDS],
                                    ['saliency_benign_y'])

    acid_cam_adv_x, acid_cam_succeed, acid_cam_cam = extract_cam_acid()
    acid_rts_adv_x, acid_rts_succeed, acid_rts_rts = extract_rts_acid()

    save_dict = locals()
    del save_dict['config']
    np.savez(config.save_path, **save_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('save_path')

    config = parser.parse_args()
    print('Please check the configuration', config)
    main(config)
