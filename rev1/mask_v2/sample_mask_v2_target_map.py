#!/usr/bin/env python
import argparse

import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, Sampler
from torchvision.transforms import ToTensor, CenterCrop, Normalize, Resize, Compose
from torchvision.models import resnet50, densenet169

from ia_utils.data_utils import ImageFolderWithPaths, imagenet_normalize
from ia_utils.model_utils import freeze_model
from expr_attacks.mask_generator import generate_mask_per_batch_v2, get_default_mask_config
from expr_attacks.mask_model import MASKV2

IMAGENET_VAL_DIR = '/home/xinyang/Datasets/imagenet_val/'
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SimpleSampler(Sampler):

    def __init__(self, indices):
        super(SimpleSampler, self).__init__(indices)
        self.indices = indices

    def __iter__(self):
        return (index for index in self.indices)

    def __len__(self):
        return len(self.indices)


def load_model(model_name, is_cuda):
    if model_name == 'resnet50':
        model = resnet50(pretrained=True)
    if model_name == 'densenet169':
        model = densenet169(pretrained=True)
    model.train(False)
    freeze_model(model)
    if is_cuda:
        model.to(torch.device('cuda:0'))
    return model


def send_data(images, labels, paths, is_cuda):
    if is_cuda:
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
    images.cuda()
    return images, labels, paths


def main(config):
    rs = np.random.RandomState(None)

    model = load_model(config.model, config.is_cuda)
    model_tup = (model, imagenet_normalize, (224, 224))

    dataset = ImageFolderWithPaths(IMAGENET_VAL_DIR, transform=Compose(
        [Resize(256), CenterCrop(224), ToTensor()]
    ))
    in_dobj = dict(np.load(config.input_path).items())
    img_yt = in_dobj['img_yt']
    df = pd.read_csv(config.ref_csv, header=0)

    indices_map = {path: index for index, (path, _) in enumerate(dataset.samples)}
    indices_per_class = [[] for _ in range(1000)]
    for i, row in df.iterrows():
        if row['pred'] == row['label']:
            indices_per_class[row['label']].append(indices_map[row['path']])

    for index, (path, _) in enumerate(dataset.samples):
        indices_map[path] = index
    target_indices = []
    for i in range(len(img_yt)):
        yt = img_yt[i]
        target_indices.append(rs.choice(indices_per_class[yt]))
    sampler = SimpleSampler(target_indices)
    loader = DataLoader(dataset, batch_size=40, sampler=sampler, pin_memory=True)
    mask_model = MASKV2(config.is_cuda)
    mask_config = get_default_mask_config()
    mask_config.update(l1_lambda=1e-4, tv_lambda=1e-2, n_iters=300)
    target_images, target_masks = [], []
    for batch in loader:
        images, labels, paths = batch
        target_images.append(images.numpy())
        batch = send_data(images, labels, paths, config.is_cuda)
        images, labels = batch[:2]
        mask = generate_mask_per_batch_v2(mask_config, mask_model, model_tup, (images, labels), config.is_cuda)
        target_masks.append(mask.detach().cpu().numpy())
    target_masks = np.concatenate(target_masks)
    target_images = np.concatenate(target_images)
    in_dobj['img_yt'] = img_yt
    in_dobj['target_mask'] = target_masks
    in_dobj['target_images'] = target_images
    np.savez(config.output_path, **in_dobj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('model')
    parser.add_argument('ref_csv')
    parser.add_argument('output_path')
    parser.add_argument('-g', '--gpu', dest='is_cuda', action='store_true')

    main(parser.parse_args())
