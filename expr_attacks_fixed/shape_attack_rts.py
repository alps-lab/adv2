#!/usr/bin/env python
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd as autograd

from expr_attacks.rts_model_resnet50 import RTSResnet50
from expr_attacks.commons import RTS_RESNET50_CKPT_DIR
from expr_attacks.commons import PGD_SAVE_PERIOD
from expr_attacks.utils import freeze_model


def attack_batch(config, rts_model, batch_tup, rts_benign):
    cuda = config.device == 'gpu'
    bx_np, by_np = batch_tup
    batch_size = len(bx_np)
    bx, by = torch.tensor(bx_np), torch.tensor(by_np)
    m0 = torch.tensor(rts_benign)
    if cuda:
        bx, by, m0 = bx.cuda(), by.cuda(), m0.cuda()
    bx_adv = bx.clone().detach().requires_grad_()

    s1_lr = config.s1_lr
    s2_lr = config.s2_lr
    eps = config.epsilon

    dobj = {}

    for i in range(config.s1_iters):
        logits = rts_model.blackbox_logits_fn(bx_adv)
        loss = F.nll_loss(F.log_softmax(logits, dim=-1), by, reduction='sum')
        loss_grad = autograd.grad([loss], [bx_adv])[0]

        # first record message
        if i % 100 == 0:
            with torch.no_grad():
                loss_adv_mu = np.asscalar(loss) / batch_size
                pred = torch.max(logits, 1)[1]
                num_succeed = np.asscalar(torch.sum(by == pred))
            print('s1-step: %d, loss adv: %.2f, succeed: %d' % (i, loss_adv_mu, num_succeed))

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
        rts, rts_logits = rts_model.saliency_fn(bx_adv, by, model_confidence=6, return_classification_logits=True)
        diff = rts - m0
        loss_rts = torch.sum((diff * diff).view(batch_size, -1).mean(1))
        logits = rts_model.blackbox_logits_fn(bx_adv)
        loss_adv = F.nll_loss(F.log_softmax(logits, dim=-1), by, reduction='sum')
        rts_adv_loss = F.nll_loss(F.log_softmax(rts_logits, dim=-1), by, reduction='sum')
        loss = 0.1 * rts_adv_loss + loss_adv + c_now * loss_rts
        loss_grad = autograd.grad([loss], [bx_adv])[0]

        if i % 100 == 0:
            with torch.no_grad():
                pred = torch.argmax(logits, 1)
                loss_rts_mu = np.asscalar(loss_rts) / batch_size
                loss_adv_mu = np.asscalar(loss_adv) / batch_size
                num_succeed = np.asscalar(torch.sum(by == pred))
                loss_adv = loss_adv_mu
                loss_rts = loss_rts_mu
            print('s2-step: %d, loss adv: %.2f, loss rts: %.5f, succeed: %d' % (i, loss_adv, loss_rts, num_succeed))

        # update
        with torch.no_grad():
            loss_grad_sign = loss_grad.sign()
            bx_adv.data.add_(-s2_lr, loss_grad_sign)
            diff = bx_adv - bx
            diff.clamp_(-eps, eps)
            bx_adv.data = diff + bx
            bx_adv.data.clamp_(0, 1)
        del loss_grad

    rts = rts_model.saliency_fn(bx_adv, by, model_confidence=6, return_classification_logits=False)
    logits = rts_model.blackbox_logits_fn(bx_adv)

    dobj['adv_x' ] = bx_adv.detach().cpu().numpy()
    dobj['adv_rts'] = rts.detach().cpu().numpy()
    dobj['adv_logits'] = logits.detach().cpu().numpy()
    dobj['adv_succeed'] = (logits.argmax(1) == by).detach().cpu().numpy().astype(np.int64)
    dobj['trts'] = rts_benign
    return dobj


def attack(config):
    rts_model = RTSResnet50(RTS_RESNET50_CKPT_DIR, config.device == 'gpu')
    freeze_model(rts_model.blackbox_model)
    freeze_model(rts_model.saliency)

    data_arx = np.load(config.data_path)
    img_x, img_yt = data_arx['att_imgs'], data_arx['att_yts']
    rts_target = data_arx['att_trts']
    rts_benign = data_arx['att_brts']

    n, batch_size = len(img_x), config.batch_size
    num_batches = (n + batch_size - 1) // batch_size
    save_dobjs = []
    for i in range(num_batches):
        si = i * batch_size
        ei = min(si + batch_size, n)
        bx, byt, bm0 = img_x[si:ei], img_yt[si:ei], rts_target[si:ei]
        dobj = attack_batch(config, rts_model, (bx, byt), bm0)
        dobj['brts'] = rts_benign[si:ei]
        dobj['img_x'] = bx
        save_dobjs.append(dobj)

    keys = list(save_dobjs[0].keys())
    save_dobj = {}
    for key in keys:
        save_dobj[key] = np.concatenate([i[key] for i in save_dobjs], axis=0)
    np.savez(config.save_path, **save_dobj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('save_path')
    parser.add_argument('-d', '--device', dest='device', choices=['cpu', 'gpu'])
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=40)
    parser.add_argument('-e', '--epsilon', dest='epsilon', type=float, default=0.031)
    parser.add_argument('--s1-iters', dest='s1_iters', type=int, default=300)
    parser.add_argument('--s1-lr', dest='s1_lr', type=float, default=1./255)
    parser.add_argument('--s2-iters', dest='s2_iters', type=int, default=1000)
    parser.add_argument('--s2-lr', dest='s2_lr', type=float, default=1./255)
    parser.add_argument('-c', dest='c', type=float, default=5.)
    config = parser.parse_args()

    if config.device is None:
        if torch.cuda.is_available():
            config.device = 'gpu'
        else:
            config.device = 'cpu'
    print('Please check the configuration', config)
    attack(config)
