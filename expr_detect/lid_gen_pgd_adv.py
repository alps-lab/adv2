#!/usr/bin/env python
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd as autograd

from expr_detect.commons import LID_DATA_PATH, LID_ADV_DATA_PATH
from expr_detect.lid_encoders import resnet50encoder
from ia_utils.data_utils import imagenet_normalize
from expr_attacks.utils import freeze_model


def load_model(config):
    encoder = resnet50encoder(True)
    encoder.train(False)
    freeze_model(encoder)
    if config.device == 'gpu':
        encoder.cuda()

    return encoder, imagenet_normalize, (224, 224)


def vanilla_attack_batch(config, model_tup, batch_tup):
    model, pre_fn, shape = model_tup
    bx_np, byt_np = batch_tup

    dobj = {}
    batch_size = len(bx_np)
    bx, byt = torch.tensor(bx_np), torch.tensor(byt_np)
    best_loss = torch.zeros(batch_size)
    if config.device == 'gpu':
        bx, byt = bx.cuda(), byt.cuda()
        best_loss = best_loss.cuda()
    bx_adv = bx.clone().detach()
    bx_adv.requires_grad = True
    bx_adv_found = bx.clone().detach()

    for j in range(config.steps):
        logits = model(pre_fn(bx_adv))[-1]
        loss = F.nll_loss(logits, byt, reduction='none')
        if j % 250 == 0:
            print('step: %d, average loss: %.4f' % (j, np.asscalar(loss.mean())))
        pgd_grad = autograd.grad([loss.sum()], [bx_adv])[0]
        mask = torch.nonzero(loss < best_loss)[:, 0]
        bx_adv_found.data[mask] = bx_adv.data[mask]
        with torch.no_grad():
            bx_adv_new = bx_adv - config.alpha * pgd_grad.sign()
            diff = bx_adv_new - bx
            diff.clamp_(-config.epsilon, config.epsilon)
            bx_adv.data = bx.data + diff
            bx_adv.data.clamp_(0, 1)

    with torch.no_grad():
        logits_adv = model(pre_fn(bx_adv_found))[-1]
    logits_adv = logits_adv.cpu().numpy()
    dobj['logits_x'] = logits_adv
    dobj['adv_x'] = bx_adv_found.detach().cpu().numpy()
    dobj['succeed'] = (np.argmax(logits_adv, -1) == byt_np).astype(np.int64)
    dobj['yt'] = byt_np
    return dobj


def generate_pgd_adv_for_fold(config, fold_num, model_tup):
    data_arx = np.load(os.path.join(config.data_dir, LID_DATA_PATH % fold_num))
    img_x, img_yt = data_arx['img_x'], data_arx['img_yt']

    n, batch_size = len(img_x), config.batch_size
    num_batches = (n + batch_size - 1) // batch_size
    batch_dobjs = []
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min(n, start_index + batch_size)
        bx_np, byt_np = (img_x[start_index:end_index], img_yt[start_index:end_index])
        robj = vanilla_attack_batch(config, model_tup, (bx_np, byt_np))
        batch_dobjs.append(robj)
    keys = batch_dobjs[0].keys()
    robj = {}
    for key in keys:
        robj[key] = np.concatenate([dobj[key] for dobj in batch_dobjs], axis=0)
    np.savez(os.path.join(config.save_dir, LID_ADV_DATA_PATH % fold_num), **robj)


def main(config):
    model_tup = load_model(config)
    fold_start, fold_end = config.fold_start, config.fold_end
    assert fold_start <= fold_end
    fold_i = fold_start
    os.makedirs(config.save_dir, exist_ok=True)
    while fold_i <= fold_end:
        print('generating regular PGD ADV for fold %d.' % fold_i)
        generate_pgd_adv_for_fold(config, fold_i, model_tup)
        print('generated regular PGD ADV for fold %d,' % fold_i)
        fold_i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('save_dir')
    parser.add_argument('fold_start', type=int)
    parser.add_argument('fold_end', type=int)
    parser.add_argument('-b', '--batch-size', type=int, dest='batch_size', default=40)
    parser.add_argument('-d', '--device', choices=['cpu', 'gpu'], dest='device')
    parser.add_argument('-s', '--steps', type=int, dest='steps', default=1100)
    parser.add_argument('--alpha', dest='alpha', type=float, default=1./255)
    parser.add_argument('-e', '--epsilon', dest='epsilon', type=float, default=0.031)

    config = parser.parse_args()
    if config.device is None:
        if torch.cuda.is_available():
            config.device = 'gpu'
        else:
            config.device = 'cpu'
    print('Please check the configuration', config)
    main(config)
