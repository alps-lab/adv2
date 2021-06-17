#!/usr/bin/env python
import argparse
import os
from collections import OrderedDict
from itertools import product

import numpy as np


BENIGN_DIR = '/home/xinyang/Data/intattack/fixup1/resnet_data/'
BENIGN_GS_DIR = '/home/xinyang/Data/intattack/rev2/resnet_gs/'

BENIGN_FMT = 'fold_%d.npz'
BENIGN_GS_FMT = 'fold_%d.npz'

ACID_GS_DIR = '/home/xinyang/Data/intattack/rev2/resnet_acid_gs_v2/'

FOLDS = [1, 2, 8]


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


def extract_gs_acid():
    adv_template = 'best_adv'
    adv_gs_template = 'best_adv_gs'

    path_tups = product([os.path.join(ACID_GS_DIR, 'fold%d_' % i) for i in FOLDS],
                    ['b%d.npz' % (i + 1) for i in range(10)])
    paths = [tup[0] + tup[1] for tup in path_tups]
    print(paths)
    adv_x, adv_gs = extract_and_concat(
        paths,
        [adv_template, adv_gs_template])
    succeed = np.ones(len(adv_x), dtype=np.int64)
    return adv_x, succeed, adv_gs


def main(config):
    img_x, img_y, img_yt = extract_and_concat([os.path.join(BENIGN_DIR, BENIGN_FMT % fold) for fold in FOLDS],
                                              ['img_x', 'img_y', 'img_yt'])
    benign_gs, = extract_and_concat([os.path.join(BENIGN_GS_DIR, BENIGN_GS_FMT % fold) for fold in FOLDS],
                                        ['benign_gs'])

    acid_gs_adv_x, acid_gs_succeed, acid_gs_gs = extract_gs_acid()

    save_dict = locals()
    del save_dict['config']
    np.savez(config.save_path, **save_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('save_path')

    config = parser.parse_args()
    print('Please check the configuration', config)
    main(config)
