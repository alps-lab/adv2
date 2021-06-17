#!/usr/bin/env python
import argparse

import tqdm
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, CenterCrop, Normalize, Resize, Compose
from torchvision.models import resnet50, densenet169

from ia_utils.data_utils import ImageFolderWithPaths
from ia_utils.model_utils import freeze_model


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
    dataset = ImageFolderWithPaths(config.data_dir, transform=Compose([
                                                                Resize(256),
                                                                CenterCrop(224),
                                                                ToTensor(),
                                                                Normalize(IMAGENET_MEAN, IMAGENET_STD)
                                                                ]))
    loader = DataLoader(dataset, batch_size=128, shuffle=False, pin_memory=True, num_workers=4)
    model = load_model(config.model, config.is_cuda)

    r_paths, r_labels, r_preds = [[] for _ in range(3)]
    for images, labels, paths in tqdm.tqdm(loader):
        images, labels, paths = send_data(images, labels, paths, config.is_cuda)
        logits = model(images)
        preds = logits.argmax(1)

        r_paths.extend(paths)
        r_labels.extend(labels.tolist())
        r_preds.extend(preds.tolist())

    df = pd.DataFrame.from_dict(dict(path=r_paths, label=r_labels, pred=r_preds)).set_index('path')
    df.to_csv(config.save_path)
    print('accuracy:', (df['label'] == df['pred']).values.mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('model', choices=['resnet50', 'densenet169'])
    parser.add_argument('save_path')
    parser.add_argument('-g', '--gpu', action='store_true', dest='is_cuda')

    main(parser.parse_args())
