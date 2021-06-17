#!/usr/bin/env python
import os
import argparse

import numpy as np
import torch
import torch.nn as nn

from expr_detect.test_lid_classifier import test_lid_model
from expr_detect.lid_utilities import load_pretrained_model, load_train_data
from expr_detect.lid_encoders import resnet50encoder
from ia_utils.data_utils import imagenet_normalize
from expr_attacks.utils import freeze_model


def main(config):
    adv_x_template = 'pgd_step_%d_adv_x'
    adv_succeed_template = 'pgd_step_%d_succeed'
    scalar, lr_model, k = load_pretrained_model(config.model)
    model = resnet50encoder(True)
    model.train(False)
    freeze_model(model)

    train_x = load_train_data()
    all_test_x = []
    all_test_y = []

    for fold_num in config.folds:
        benign_img = np.load(os.path.join(config.benign_dir, 'fold_%d.npz' % fold_num))['img_x']
        noisy_img = np.maximum(0, np.minimum(1, benign_img +
                                             np.random.uniform(-0.031, 0.031,
                                                               size=benign_img.shape).astype(np.float32)))
        adv_arx = np.load(os.path.join(config.adv_dir, 'mask_attack_pgd_vanilla_fold_resnet_%d.npz' % fold_num))
        adv_img = adv_arx[adv_x_template % config.adv_step]
        adv_succeed = adv_arx[adv_succeed_template % config.adv_step].astype(np.bool)
        benign_img = benign_img[adv_succeed]
        noisy_img = noisy_img[adv_succeed]
        adv_img = adv_img[adv_succeed]

        n = np.count_nonzero(adv_succeed)
        test_x = np.concatenate([benign_img, noisy_img, adv_img], axis=0)
        test_y = np.concatenate([np.zeros(n), np.zeros(n), np.ones(n)], axis=0)
        all_test_x.append(test_x)
        all_test_y.append(test_y)
    all_test_x = np.concatenate(all_test_x, axis=0)
    all_test_y = np.concatenate(all_test_y, axis=0)

    cuda = config.device == 'gpu'
    if cuda:
        model.cuda()
    auc_roc_score, confuse_mat = test_lid_model(
        (model, imagenet_normalize), (scalar, lr_model), train_x, all_test_x, all_test_y, k, cuda)
    print('auc_roc_score:', auc_roc_score)
    print('confusion matrix:', confuse_mat)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['reg_pgd', 'acid_rts'])
    parser.add_argument('benign_dir')
    parser.add_argument('adv_dir')
    parser.add_argument('adv_step', type=int)
    parser.add_argument('-d', '--device', choices=['cpu', 'gpu'])
    parser.add_argument('--folds', nargs='+', type=int, default=[1, 2, 3])

    config = parser.parse_args()
    if config.device is None:
        if torch.cuda.is_available():
            config.device = 'gpu'
        else:
            config.device = 'cpu'
    print('Please check the configuration', config)
    main(config)
