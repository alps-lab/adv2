import os
import csv
import pickle
import argparse

import torch

from torchvision.models.resnet import resnet50
from torchvision.models.densenet import densenet169
from itertools import product

from ia_utils.data_utils import imagenet_normalize
from expr_attacks.utils import freeze_model
from expr_detect.wrapper import Wrapper, get_batch_stats
from expr_detect.commons import *


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
    model, pre_fn = load_model(config)[:2]
    data_arx = np.load(config.data_path)
    train_x, train_y = data_arx['train_benign_x'], data_arx['train_benign_y']
    wrapper = Wrapper(model, pre_fn, train_x, train_y, config.batch_size, config.device == 'gpu')
    results = {}
    tup = slice(2 if config.debug else None)
    for c1, c2, c3 in product(BIT_DEPTHS[tup], MEDIAN_SMOOTHING[tup], NONLOCAL_MEAN[tup]):
        key = (c1, c2, c3)
        print(key)
        dists = []
        for i in range(wrapper.n_batches):
            dists.append(get_batch_stats(wrapper, i, c1, c2, c3))
        results[key] = np.concatenate(dists, axis=0)
    pickle.dump(results, open(config.save_path, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['resnet50', 'densenet169'])
    parser.add_argument('data_path')
    parser.add_argument('save_path')
    parser.add_argument('-b', '--batch-size', type=int, dest='batch_size', default=80)
    parser.add_argument('-d', '--device', choices=['cpu', 'gpu'], dest='device')
    parser.add_argument('--debug', action='store_true', dest='debug')

    config = parser.parse_args()
    if config.device is None:
        if torch.cuda.is_available():
            config.device = 'gpu'
        else:
            config.device = 'cpu'
    print('Please check the configuration', config)
    main(config)
