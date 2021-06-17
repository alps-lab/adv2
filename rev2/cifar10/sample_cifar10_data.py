#!/usr/bin/env python
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision

from rev2.cifar10.data_utils import cifar10_normalize
from rev2.cifar10.model_utils import resnet50, CIFAR10_RESNET50_CKPT_PATH


def load_model():
    model = resnet50()
    ckpt_dict = torch.load(CIFAR10_RESNET50_CKPT_PATH, lambda storage, loc: storage)['net']
    nn.DataParallel(model).load_state_dict(ckpt_dict)
    model.to('cuda')
    model.train(False)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('save_dir')

    save_dir = parser.parse_args().save_dir
    os.makedirs(save_dir, exist_ok=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=torchvision.transforms.ToTensor())
    loader = DataLoader(testset, shuffle=True, batch_size=100)
    model = load_model()
    bx_nps, by_nps = [], []

    for i, (bx, by) in enumerate(loader):
        bx_np, by_np = bx.numpy(), by.numpy()
        bx, by = [t.to('cuda') for t in (bx, by)]
        bx = cifar10_normalize(bx)
        bp_np = model(bx).argmax(1).to('cpu').numpy()
        succeed = by_np == bp_np
        bx_nps.append(bx_np[succeed])
        by_nps.append(by_np[succeed])

    bx_np = np.concatenate(bx_nps)
    by_np = np.concatenate(by_nps)
    byt = np.random.randint(1, 10, len(by_np)).astype(np.int64)
    byt = np.mod(by_np + byt, 10)

    num_folds = min(len(bx_np) // 100, 5)
    print(len(bx_np), by_np, byt)

    for i in range(num_folds):
        si, ei = i * 100, (i + 1) * 100
        sl = slice(si, ei)
        np.savez(os.path.join(save_dir, 'fold_%d.npz' % (i + 1)), img_x=bx_np[sl],
                 img_y=by_np[sl], img_yt=byt[sl])
