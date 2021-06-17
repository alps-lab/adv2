#!/usr/bin/env python
import os
import argparse

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F


from expr_attacks.rts_model_resnet50 import RTSResnet50
from expr_attacks.rts_generator import get_default_rts_config
from expr_detect.commons import LID_DATA_PATH, LID_ADV_DATA_PATH


def acid_rts_attack_batch(config, rts_model, batch_tup):
    cuda = config.device == 'gpu'
    bx_np, by_np, byt_np = batch_tup
    batch_size = len(bx_np)
    bx, by, byt = torch.tensor(bx_np), torch.tensor(by_np), torch.tensor(by_np)
    if cuda:
        bx, by, byt = bx.cuda(), by.cuda(), byt.cuda()

    m0 = rts_model.saliency_fn(bx, by, model_confidence=6, return_classification_logits=False).detach()
    bx_adv = bx.clone().detach().requires_grad_()
    s1_lr = config.s1_lr
    s2_lr = config.s2_lr
    eps = config.epsilon
    dobj = {}

    for i in range(config.s1_iters):
        logits = rts_model.blackbox_logits_fn(bx_adv)
        loss = F.nll_loss(logits, byt, reduction='sum')
        loss_grad = autograd.grad([loss], [bx_adv])[0]

        if i % 100 == 0:
            print('s1-step: %d, average adv loss: %.4f' % (i, np.asscalar(loss / batch_size)))

        # then update
        with torch.no_grad():
            loss_grad_sign = loss_grad.sign()
            bx_adv.data.add_(-s1_lr, loss_grad_sign)
            diff = bx_adv - bx
            diff.clamp_(-eps, eps)
            bx_adv.data = diff + bx
            bx_adv.data.clamp_(0, 1)

    c_begin, c_final = config.c, config.c * 2
    c_inc = (c_final - c_begin) / config.s2_iters
    c_now = config.c
    for i in range(config.s2_iters):
        c_now += c_inc
        rts, rts_logits = rts_model.saliency_fn(bx_adv, byt, model_confidence=6, return_classification_logits=True)
        diff = rts - m0
        loss_rts = torch.sum((diff * diff).view(batch_size, -1).mean(1))
        logits = rts_model.blackbox_logits_fn(bx_adv)
        loss_adv = F.nll_loss(logits, byt, reduction='sum')
        rts_adv_loss = F.nll_loss(rts_logits, byt, reduction='sum')
        loss = torch.add(0.4 * rts_adv_loss + loss_adv, c_now, loss_rts)
        loss_grad = autograd.grad([loss], [bx_adv])[0]

        if i % 100 == 0:
            print('s2-step: %d, average adv loss: %.4f, average rts loss: %.4f' %
                  (i, np.asscalar(loss_adv / batch_size), np.asscalar(loss_rts / batch_size)))

        # update
        with torch.no_grad():
            loss_grad_sign = loss_grad.sign()
            bx_adv.data.add_(-s2_lr, loss_grad_sign)
            diff = bx_adv - bx
            diff.clamp_(-eps, eps)
            bx_adv.data = diff + bx
            bx_adv.data.clamp_(0, 1)

    with torch.no_grad():
        logits_adv = rts_model.blackbox_logits_fn(bx_adv)
        m_adv = rts_model.saliency_fn(bx_adv, byt, return_classification_logits=False)

    logits_adv = logits_adv.cpu().numpy()
    dobj['logits_x'] = logits_adv
    dobj['adv_x'] = bx_adv.detach().cpu().numpy()
    dobj['succeed'] = (np.argmax(logits_adv, -1) == byt_np).astype(np.int64)
    dobj['yt'] = byt_np
    dobj['y'] = by_np
    dobj['benign_rts'] = m0.cpu().numpy()
    dobj['adv_rts'] = m_adv.cpu().numpy()
    return dobj


def generate_acid_rts_adv_for_fold(config, fold_num, model_tup):
    data_arx = np.load(os.path.join(config.data_dir, LID_DATA_PATH % fold_num))
    img_x, img_y, img_yt = data_arx['img_x'], data_arx['img_y'], data_arx['img_yt']

    n, batch_size = len(img_x), config.batch_size
    num_batches = (n + batch_size - 1) // batch_size
    batch_dobjs = []
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min(n, start_index + batch_size)
        bx_np, byt_np = (img_x[start_index:end_index], img_yt[start_index:end_index])
        by_np = img_y[start_index:end_index]
        robj = acid_rts_attack_batch(config, model_tup, (bx_np, by_np, byt_np))
        batch_dobjs.append(robj)
    keys = batch_dobjs[0].keys()
    robj = {}
    for key in keys:
        robj[key] = np.concatenate([dobj[key] for dobj in batch_dobjs], axis=0)
    np.savez(os.path.join(config.save_dir, LID_ADV_DATA_PATH % fold_num), **robj)


def main(config):
    rts_config = get_default_rts_config('resnet50')
    rts_model = RTSResnet50(rts_config['ckpt_dir'], config.device == 'gpu')
    fold_start, fold_end = config.fold_start, config.fold_end
    assert fold_start <= fold_end
    fold_i = fold_start
    os.makedirs(config.save_dir, exist_ok=True)
    while fold_i <= fold_end:
        print('generating ACID RTS ADV for fold %d.' % fold_i)
        generate_acid_rts_adv_for_fold(config, fold_i, rts_model)
        print('generated ACID RTS ADV for fold %d,' % fold_i)
        fold_i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('save_dir')
    parser.add_argument('fold_start', type=int)
    parser.add_argument('fold_end', type=int)
    parser.add_argument('-b', '--batch-size', type=int, dest='batch_size', default=40)
    parser.add_argument('-d', '--device', choices=['cpu', 'gpu'], dest='device')
    parser.add_argument('--s1-iters', dest='s1_iters', type=int, default=300)
    parser.add_argument('--s1-lr', dest='s1_lr', type=float, default=1./255)
    parser.add_argument('--s2-iters', dest='s2_iters', type=int, default=1000)
    parser.add_argument('--s2-lr', dest='s2_lr', type=float, default=1./255)
    parser.add_argument('-c', dest='c', type=float, default=1680)
    parser.add_argument('-e', '--epsilon', dest='epsilon', type=float, default=0.031)

    config = parser.parse_args()
    if config.device is None:
        if torch.cuda.is_available():
            config.device = 'gpu'
        else:
            config.device = 'cpu'
    print('Please check the configuration', config)
    main(config)
