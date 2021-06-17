#!/usr/bin/env python
import os
import argparse

import numpy as np
import torch

from torchvision.models.resnet import resnet50
from torchvision.models.vgg import vgg16
from torchvision.models.densenet import densenet169

from ia_utils.data_utils import imagenet_normalize
from expr_attacks.utils import load_images, load_images_list


def load_model(config):
    pre_fn = imagenet_normalize
    if config.model == 'vgg16':
        model = vgg16(pretrained=True)
        shape = (224, 224)
    if config.model == 'resnet50':
        model = resnet50(pretrained=True)
        shape = (224, 224)
    if config.model == 'densenet169':
        model = densenet169(pretrained=True)
        shape = (224, 224)
    model.train(False)
    if config.device == 'gpu':
        model.cuda()
    return model, pre_fn, shape


def get_attack_list(config, model_tup, images_tup):
    model, pre_fn, shape = model_tup
    image_paths, image_labels = images_tup
    sub_images, sub_labels, sub_image_paths = [], [], []
    tot_attempt, tot_accepted = 0, 0
    n = len(image_paths)
    num_batches = (n + config.batch_size - 1) // config.batch_size
    for i in range(num_batches):
        start_index = i * config.batch_size
        end_index = min(start_index + config.batch_size, n)
        if tot_accepted >= config.total:
            break
        bpath = image_paths[start_index:end_index]
        bx_np = load_images(bpath, shape=shape)
        by_np = image_labels[start_index:end_index]
        bx, by = torch.tensor(bx_np), torch.tensor(by_np)
        if config.device == 'gpu':
            bx, by = bx.cuda(), by.cuda()
        with torch.no_grad():
            b_logits = model(pre_fn(bx))
            b_pred = torch.max(b_logits, 1)[1].cpu().numpy()
        for i in range(end_index - start_index):
            tot_attempt += 1
            if b_pred[i] == by_np[i]:
                tot_accepted += 1
                sub_images.append(bx_np[i])
                sub_labels.append(by_np[i])
                sub_image_paths.append(bpath[i])

    sub_images = np.stack(sub_images)[:config.total]
    sub_labels = np.stack(sub_labels).astype(np.int64)[:config.total]
    sub_targets = (sub_labels + np.random.RandomState(config.seed).randint(1, 1000,  len(sub_labels))) % 1000
    sub_image_paths = np.stack(sub_image_paths)[:config.total]

    return sub_images, sub_labels, sub_targets, sub_image_paths


def main(config):
    model, pre_fn, shape = load_model(config)
    paths, labels = load_images_list(config.csv_path)
    img_x, img_y, img_yt, img_path = get_attack_list(config, (model, pre_fn, shape), (paths, labels))
    assert np.count_nonzero(img_y == img_yt) == 0
    print('selected: %d' % len(img_x))
    num_folds = (len(img_x) + 100 - 1) // 100
    os.makedirs(config.save_dir, exist_ok=True)
    for fold in range(num_folds):
        start_index = fold * 100
        end_index = min(len(img_x), start_index + 100)
        tup = slice(start_index, end_index)
        fx, fy, fty, fp = img_x[tup], img_y[tup], img_yt[tup], img_path[tup]
        np.savez(os.path.join(config.save_dir, 'fold_%d.npz' % (fold + 1)),
                              img_x=fx, img_y=fy, img_yt=fty, img_path=fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', help='csv path to list of images to attack')
    parser.add_argument('model', choices=['vgg16', 'resnet50', 'densenet169'])
    parser.add_argument('save_dir', help='dir to save')
    parser.add_argument('-b', '--batch-size', type=int, dest='batch_size', default=40)
    parser.add_argument('-t', '--total', dest='total', type=int, default=1000)
    parser.add_argument('-d', '--device', choices=['cpu', 'gpu'], dest='device')
    parser.add_argument('-s', '--seed', type=int, dest='seed', default=999888777)

    config = parser.parse_args()
    if config.device is None:
        if torch.cuda.is_available():
            config.device = 'gpu'
        else:
            config.device = 'cpu'
    print('Please check the configuration', config)
    main(config)
