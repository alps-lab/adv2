#!/usr/bin/env python
import argparse
import os
import pickle

import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, confusion_matrix

from expr_attacks.utils import freeze_model
from ia_utils.data_utils import imagenet_normalize
from expr_detect.commons import LID_ADV_DATA_PATH, LID_DATA_PATH
from expr_detect.lid_encoders import resnet50encoder

from expr_detect.lid_utilities import extract_lid_features, get_big_batch_features
from expr_detect.test_lid_classifier import test_lid_model


K_GRIDS = [10, 20, 30, 40, 50]


def load_model(config):
    cuda = config.device == 'gpu'
    model = resnet50encoder(True)
    model.train(False)
    freeze_model(model)
    if cuda:
        model.cuda()
    return model, imagenet_normalize


def read_data(config):
    benign_imgs = []
    adv_imgs = []
    for fold_i in range(config.fold_start, config.fold_end + 1):
        benign_imgs.append(np.load(os.path.join(config.benign_dir,LID_DATA_PATH % fold_i))['img_x'])
        adv_imgs.append(np.load(os.path.join(config.adv_dir, LID_ADV_DATA_PATH % fold_i))['adv_x'])
    benign_imgs = np.concatenate(benign_imgs, axis=0)
    adv_imgs = np.concatenate(adv_imgs, axis=0)
    return benign_imgs, adv_imgs


def train_lid_model(model_tup, train_x, train_adv, k, cuda):
    n_features = 20
    n, batch_size = len(train_x), 100
    n_batches = (n + batch_size - 1) // batch_size

    lr_train_x = []
    lr_train_y = []
    mle_func = lambda v: -k / np.sum(np.log(v / v[-1]))

    for i in range(n_batches):
        si = i * batch_size
        ei = min(n, si + batch_size)
        if ei - si < batch_size:
            break
        bx_np = train_x[si:ei]
        features = get_big_batch_features(model_tup, bx_np, n_features, cuda)
        bx_noisy_np = train_x[si:ei] + np.random.uniform(-0.031, 0.031, size=bx_np.shape).astype(np.float32)
        features_noisy = get_big_batch_features(model_tup, bx_noisy_np, n_features, cuda)
        bx_adv_np = train_adv[si:ei]
        features_adv = get_big_batch_features(model_tup, bx_adv_np, n_features, cuda)

        b_dist_mats = [f[:, 1:k+1] for f in extract_lid_features(features, features)]
        b_dist_mats_noisy = [f[:, 1:k+1] for f in extract_lid_features(features, features_noisy)]
        b_dist_mats_adv = [f[:, 1:k+1] for f in extract_lid_features(features, features_adv)]
        blid_mle = np.concatenate([np.apply_along_axis(mle_func, 1, f)[:, None] for f in b_dist_mats], axis=1)
        blid_mle_noisy = np.concatenate([np.apply_along_axis(mle_func, 1, f)[:, None] for f in b_dist_mats_noisy], axis=1)
        blid_mle_adv = np.concatenate([np.apply_along_axis(mle_func, 1, f)[:,  None] for f in b_dist_mats_adv], axis=1)

        lr_train_x.append(blid_mle)
        lr_train_y.append(np.zeros(batch_size, np.int64))
        lr_train_x.append(blid_mle_noisy)
        lr_train_y.append(np.zeros(batch_size, np.int64))
        lr_train_x.append(blid_mle_adv)
        lr_train_y.append(np.ones(batch_size, np.int64))

    lr_train_x = np.concatenate(lr_train_x, axis=0)
    lr_train_y = np.concatenate(lr_train_y, axis=0)
    scalar = MinMaxScaler().fit(lr_train_x)
    lr_model = LogisticRegressionCV(n_jobs=-1)
    lr_model.fit(scalar.transform(lr_train_x), lr_train_y)
    return scalar, lr_model


def create_test_set(test_x, test_adv):
    noise = np.random.uniform(-0.031, 0.031, size=test_x.shape).astype(np.float32)
    test_noisy_x = np.maximum(0, np.minimum(1, test_x + noise))

    lr_test_x = np.concatenate([test_x, test_noisy_x, test_adv], axis=0)
    lr_test_y = np.concatenate([np.zeros(2 * len(test_x), np.int64), np.ones(len(test_adv), np.int64)], axis=0)
    return lr_test_x, lr_test_y


def main(config):
    model_tup = load_model(config)
    benign_imgs, adv_imgs = read_data(config)
    indices = np.random.RandomState(77885544).choice(len(benign_imgs), size=len(benign_imgs), replace=False)
    kf = KFold(5, True)
    best_score = -np.inf
    best_lr_model = None
    for kid, (train_index, val_index) in enumerate(kf.split(indices)):
        print(train_index.shape, val_index.shape)
        train_index = indices[train_index]
        val_index = indices[val_index]

        bx_train_np = benign_imgs[train_index]
        badv_train_np = adv_imgs[train_index]
        bx_val_np = benign_imgs[val_index]
        badv_val_np = adv_imgs[val_index]

        scalar, model = train_lid_model(model_tup, bx_train_np, badv_train_np, K_GRIDS[kid], config.device == 'gpu')
        test_x, test_y = create_test_set(bx_val_np, badv_val_np)
        test_scores = test_lid_model(model_tup, (scalar, model), bx_train_np, test_x, test_y, K_GRIDS[kid],
                                     config.device == 'gpu')
        if test_scores[0] > best_score:
            best_score = test_scores[0]
            best_lr_model = scalar, model, K_GRIDS[kid]
        print('k=%d, score: %s' % (K_GRIDS[kid], test_scores))
    pickle.dump({'scalar': best_lr_model[0], 'model': best_lr_model[1], 'k': best_lr_model[2]},
                open(config.save_path, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('benign_dir')
    parser.add_argument('adv_dir')
    parser.add_argument('fold_start', type=int)
    parser.add_argument('fold_end', type=int)
    parser.add_argument('save_path')
    parser.add_argument('-d', '--device', dest='device', choices=['cpu', 'gpu'])

    config = parser.parse_args()
    if config.device is None:
        if torch.cuda.is_available():
            config.device = 'gpu'
        else:
            config.device = 'cpu'
    print('Please check the configuration', config)
    main(config)
