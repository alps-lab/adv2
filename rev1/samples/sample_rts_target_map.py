#!/usr/bin/env python
import argparse

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Sampler
from torchvision.transforms import ToTensor, CenterCrop, Normalize, Resize, Compose

from ia_utils.data_utils import ImageFolderWithPaths, imagenet_normalize
from ia_utils.model_utils import freeze_model
from expr_attacks.commons import RTS_RESNET50_CKPT_DIR, RTS_DENSENET169_CKPT_DIR
from expr_attacks.rts_model_resnet50 import RTSResnet50
from expr_attacks.rts_model_densenet169 import RTSDensenet169


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


def send_data(images, labels, paths, is_cuda):
    if is_cuda:
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
    images.cuda()
    return images, labels, paths


def main(config):
    rs = np.random.RandomState(None)
    if config.model == 'resnet50':
        rts_model = RTSResnet50(RTS_RESNET50_CKPT_DIR, config.is_cuda)
    if config.model == 'densenet169':
        rts_model = RTSDensenet169(RTS_DENSENET169_CKPT_DIR, config.is_cuda)
        freeze_model(rts_model.blackbox_model)
    freeze_model(rts_model.saliency)

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
    loader = DataLoader(dataset, batch_size=64, sampler=sampler, pin_memory=True)
    target_images, target_rts = [], []

    for batch in loader:
        images, labels, paths = batch
        target_images.append(images.numpy())
        images, labels = batch[:2]
        images, labels = send_data(images, labels, paths, config.is_cuda)[:2]
        rts = rts_model.saliency_fn(images, labels, model_confidence=0.,
                                    return_classification_logits=False)
        target_rts.append(rts)
    target_rts = np.concatenate(target_rts)
    target_images = np.concatenate(target_images)
    in_dobj['img_yt'] = img_yt
    in_dobj['target_rts'] = target_rts
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
