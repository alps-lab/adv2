#!/usr/bin/env python
import pickle
import re
import argparse
from itertools import product

import torch
from torchvision.models import resnet50
from torchvision.models import densenet169

from expr_detect.wrapper import Wrapper, get_batch_stats, compute_threshold
from expr_detect.commons import *
from ia_utils.data_utils import imagenet_normalize
from expr_attacks.utils import freeze_model


def load_model(config):
    if config.model == 'resnet50':
        model = resnet50(pretrained=True)
    if config.model == 'densenet169':
        model = densenet169(pretrained=True)
    if config.device == 'gpu':
        model.cuda()
    model.eval()
    freeze_model(model)
    pre_fn = imagenet_normalize
    shape = (224, 224)
    return model, pre_fn, shape


def main(config):
    adv_x_step_pattern = 'pgd_s2_step_%d_adv_x'
    adv_succeed_step_pattern = 'pgd_s2_step_%d_adv_succeed'

    target_step = 1200
    if config.model == 'resnet50':
        trained_model = pickle.load(open(RESNET50_MODEL_PATH, 'rb'))
    if config.model == 'densenet169':
        trained_model = pickle.load(open(DENSENET169_MODEL_PATH, 'rb'))
    model, pre_fn = load_model(config)[:2]
    thresholds = compute_threshold(trained_model, config.threshold)

    succeed = []
    train_x = []
    for path in config.data_paths:
        pgd_arx = np.load(path)
        succeed.append(pgd_arx[adv_succeed_step_pattern % target_step])
        train_x.append(pgd_arx[adv_x_step_pattern % target_step])
    succeed, train_x = np.concatenate(succeed, axis=0), np.concatenate(train_x, axis=0)

    final_results = {}
    n = len(train_x)
    wrapper = Wrapper(model, pre_fn, train_x, None, config.batch_size, config.device == 'gpu')
    for c1, c2, c3 in product(BIT_DEPTHS, MEDIAN_SMOOTHING, NONLOCAL_MEAN):
        recorder = Recorder()
        key = (c1, c2, c3)
        if all([c is None for c in key]):
            continue
        dists = []
        for i in range(wrapper.n_batches):
            dists.append(get_batch_stats(wrapper, i, c1, c2, c3))
        dists = np.concatenate(dists, axis=0)
        flag = (dists > thresholds[key]).astype(np.int64)
        for i in range(n):
            recorder.append(RECORDER_ADV if succeed[i] else RECORDER_FAILED_ADV, flag[i], '')
        final_results[key] = recorder.produce_stats()
        del recorder

    if config.save_path is not None:
        pickle.dump(final_results, open(config.save_path, 'wb'))
    print(final_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_paths', nargs='*')
    parser.add_argument('model', choices=['resnet50', 'densenet169'])
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=50)
    parser.add_argument('-t', '--threshold', dest='threshold', type=float, default=0.95)
    parser.add_argument('-d', '--device', choices=['cpu', 'gpu'])
    parser.add_argument('-s', '--save-path', dest='save_path')

    config = parser.parse_args()
    if config.device is None:
        if torch.cuda.is_available():
            config.device = 'gpu'
        else:
            config.device = 'cpu'
    print('Please check the configuration', config)
    main(config)
