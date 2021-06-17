#!/usr/bin/env python
import argparse
import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from rev1.isic_acid.isic_model import get_isic_model_on_resnet50
from rev1.isic_acid.isic_utils import ISIC_RESNET50_CKPT_PATH


def identity(x):
    return x


def pgd_attack(model_tup, bx, by):
    model, pre_fn = model_tup[:2]
    bx0 = bx.clone()
    bx.requires_grad_()
    for i in range(config.steps):
        bxp = pre_fn(bx)
        bl = model(bxp)
        loss = F.nll_loss(F.log_softmax(bl), by, reduction='sum')
        grad = autograd.grad([loss], [bx])[0]

        bx.data = bx.data - 1./255 * grad.sign()
        r = bx.data - bx0.data
        r.clamp_(-0.031, 0.031)
        bx.data = bx0 + r
        bx.data.clamp_(0, 1)

    with torch.no_grad():
        bxp = pre_fn(bx)
        bl = model(bxp)
        pred = bl.argmax(1)
    succeed = (pred == by).long().cpu().numpy()
    print('succeed', np.mean(succeed))
    return {'adv_x': bx.detach().cpu().numpy(), 'adv_logits': bl.cpu().numpy(),
            'adv_succeed': succeed}


def main(config):
    model = get_isic_model_on_resnet50(False, True, ckpt_path=ISIC_RESNET50_CKPT_PATH)
    model.to('cuda')
    model_tup = (model, identity)

    hobj = np.load(config.data_path)
    img_x, img_y = hobj['img_x'], hobj['img_yt']
    n, batch_size = len(img_x), config.batch_size
    n_batches = (n + batch_size - 1) // batch_size

    results = []
    for i in range(n_batches):
        si = i * batch_size
        ei = min(n, si + batch_size)
        bx_np, by_np = img_x[si:ei], img_y[si:ei]
        bx, by = [torch.tensor(t, device=config.device) for t in (bx_np, by_np)]
        result = pgd_attack(model_tup, bx, by)
        results.append(result)

    keys = results[-1].keys()
    save_dobj = {key: None for key in keys}
    for key in keys:
        save_dobj[key] = np.concatenate([result[key] for result in results])
    save_dobj['img_x'] = hobj['img_x']
    save_dobj['img_y'] = hobj['img_y']
    save_dobj['img_yt'] = hobj['img_yt']
    np.savez(config.save_path, **save_dobj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('save_path')
    parser.add_argument('-d', '--device', dest='device', choices=['cuda', 'cpu'])
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=50)
    parser.add_argument('-s', '--steps', dest='steps', type=int, default=500)

    config = parser.parse_args()
    if config.device is None:
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('configuration:', config)

    main(config)
