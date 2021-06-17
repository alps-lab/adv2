#!/usr/bin/env python
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd as autograd

from expr_attacks.rts_model_resnet50 import RTSResnet50
from expr_attacks.rts_model_densenet169 import RTSDensenet169
from expr_attacks.commons import RTS_RESNET50_CKPT_DIR, RTS_DENSENET169_CKPT_DIR
from expr_attacks.commons import PGD_SAVE_PERIOD
from expr_attacks.utils import freeze_model
from expr_detect.squeeze_utils import bit_depth_torch
from expr_detect.lid_utilities import MedianPool2d


MEDIAN_SMOOTHING_3X3_THRESH = 1.3244714
MEDIAN_SMOOTHING_2X2_THRESH = 0.96955955


def attack_batch(config, rts_model, batch_tup, rts_benign):
    cuda = config.device == 'gpu'
    bx_np, by_np, byt_np = batch_tup
    batch_size = len(bx_np)
    bx, by, byt = torch.tensor(bx_np), torch.tensor(by_np), torch.tensor(byt_np)
    m0 = torch.tensor(rts_benign)
    if cuda:
        bx, by, byt, m0 = bx.cuda(), by.cuda(), byt.cuda(), m0.cuda()
    bx_adv = bx.clone().detach().requires_grad_()

    s1_lr = config.s1_lr
    s2_lr = config.s2_lr
    eps = config.epsilon

    adv_step_template = 'pgd_s2_step_%d_adv_x'
    rts_step_template = 'pgd_s2_step_%d_adv_rts'
    logits_step_template = 'pgd_s2_step_%d_adv_logits'
    dobj = {}

    bit_depth = 3
    with torch.no_grad():
        upper_bound_bx_adv = torch.clamp(bx_adv + eps, 0, 1)
        lower_bound_bx_adv = torch.clamp(bx_adv - eps, 0, 1)
        upper_bound_squeezed = bit_depth_torch(upper_bound_bx_adv, bit_depth)
        lower_bound_squeezed = bit_depth_torch(lower_bound_bx_adv, bit_depth)
    squeezed_bx_adv = bit_depth_torch(bx, bit_depth)
    squeezed_bx_adv.requires_grad = True
    step_size = 1.0 / ((2 ** bit_depth) - 1)

    for i in range(config.s1_iters):
        logits = rts_model.blackbox_logits_fn(squeezed_bx_adv)
        loss = F.nll_loss(F.log_softmax(logits, dim=-1), by, reduction='sum')
        loss_grad = autograd.grad([loss], [squeezed_bx_adv])[0]

        # first record message
        if i % 100 == 0:
            with torch.no_grad():
                loss_adv_mu = np.asscalar(loss) / batch_size
                pred = torch.argmax(logits, 1)
                num_succeed = np.asscalar(torch.sum(by != pred))
            print('s1-step: %d, loss adv: %.2f, succeed: %d' % (i, loss_adv_mu, num_succeed))

        # then update
        with torch.no_grad():
            loss_grad_sign = loss_grad.sign()
            squeezed_bx_adv.data.add_(step_size, loss_grad_sign)
            diff = squeezed_bx_adv - bx
            squeezed_bx_adv.data = diff + bx
            squeezed_bx_adv.data = torch.min(squeezed_bx_adv, upper_bound_squeezed)
            squeezed_bx_adv.data = torch.max(squeezed_bx_adv, lower_bound_squeezed)

    probs = F.softmax(logits.detach(), -1)
    byt = probs.argmax(1)
    dobj['byt'] = byt.cpu().numpy()
    pre_probs = probs

    c_begin, c_final = config.c, config.c * 2
    c_inc = (c_final - c_begin) / config.s2_iters
    c_now = config.c
    median_pool_3x3 = MedianPool2d(3, same=True)
    median_pool_2x2 = MedianPool2d(2, same=True)
    for i in range(config.s2_iters):
        if bx_adv.grad is not None:
            bx_adv.grad.data.zero_()
        c_now += c_inc

        logits = rts_model.blackbox_logits_fn(bx_adv)
        probs = F.softmax(logits, -1)
        probs_dist = F.relu((probs - pre_probs).abs().sum(1) - 1.5924778 * 0.75).sum()
        loss_adv = F.nll_loss(F.log_softmax(logits, dim=-1), byt, reduction='sum')

        rts, rts_logits = rts_model.saliency_fn(bx_adv, byt, model_confidence=6, return_classification_logits=True)
        rts_adv_loss = F.nll_loss(F.log_softmax(rts_logits, dim=-1), byt, reduction='sum')
        diff = rts - m0
        loss_rts = torch.sum((diff * diff).view(batch_size, -1).mean(1))
        loss = loss_adv + 0.1 * rts_adv_loss + c_now * loss_rts + 50 * probs_dist
        loss.backward()

        # 3 x 3
        logits = rts_model.blackbox_logits_fn(bx_adv)
        probs = F.softmax(logits, -1)
        bx_adv_smoothed = median_pool_3x3(bx_adv)
        logits_smoothed = rts_model.blackbox_logits_fn(bx_adv_smoothed)
        probs_smoothed = F.softmax(logits_smoothed, -1)
        probs_l1 = (probs - probs_smoothed).abs().sum(1)
        fs_loss_3x3 = F.relu(probs_l1 - MEDIAN_SMOOTHING_3X3_THRESH * 0.75).sum()

        # 2 x 2
        bx_adv_smoothed = median_pool_2x2(bx_adv)
        logits_smoothed = rts_model.blackbox_logits_fn(bx_adv_smoothed)
        probs_smoothed = F.softmax(logits_smoothed, -1)
        probs_l1 = (probs - probs_smoothed).abs().sum(1)
        fs_loss_2x2 = F.relu(probs_l1 - MEDIAN_SMOOTHING_2X2_THRESH * 0.75).sum()

        loss = 10 * (0.75 * fs_loss_2x2 + 0.25 * fs_loss_3x3)
        loss.backward()

        loss_grad = bx_adv.grad.data
        # record examples
        if i % PGD_SAVE_PERIOD == PGD_SAVE_PERIOD - 1:
            dobj[adv_step_template % (i + 1)] = bx_adv.detach().cpu().numpy()
            dobj[rts_step_template % (i + 1)] = rts.detach().cpu().numpy()
            dobj[logits_step_template % (i + 1)] = logits.detach().cpu().numpy()

        # print message
        if i % 100 == 0:
            with torch.no_grad():
                pred = torch.argmax(logits, 1)
                loss_rts_mu = np.asscalar(loss_rts) / batch_size
                loss_adv_mu = np.asscalar(loss_adv) / batch_size
                loss_fs_mu = np.asscalar(probs_dist) / batch_size
                loss_fs_2x2_mu = np.asscalar(fs_loss_2x2) / batch_size
                loss_fs_3x3_mu = np.asscalar(fs_loss_3x3) / batch_size
                num_succeed = np.asscalar(torch.sum(byt == pred))
                loss_adv = loss_adv_mu
                loss_rts = loss_rts_mu
            print('s2-step: %d, loss adv: %.2f, loss rts: %.5f, loss fs: %.5f, loss fs 3x3: %.5f, loss fs 2x2: %.5f, succeed: %d' % (
                i, loss_adv, loss_rts, loss_fs_mu, loss_fs_3x3_mu, loss_fs_2x2_mu, num_succeed))

        # update
        with torch.no_grad():
            loss_grad_sign = loss_grad.sign()
            bx_adv.data.add_(-s2_lr, loss_grad_sign)
            diff = bx_adv - bx
            diff.clamp_(-eps, eps)
            bx_adv.data = diff + bx
            bx_adv.data.clamp_(0, 1)
        del loss_grad

    return dobj


def analyze_batch(config, rts_model, batch_tup, rts_benign, pre_dobj):
    dobj = {}
    rts_step_template = 'pgd_s2_step_%d_adv_rts'
    logits_step_template = 'pgd_s2_step_%d_adv_logits'
    succeed_step_template = 'pgd_s2_step_%d_adv_succeed'
    pred_step_template = 'pgd_s2_step_%d_pred'
    rts_l1_dist_step_template = 'pgd_s2_step_%d_rts_l1_dist'
    rts_l2_dist_step_template = 'pgd_s2_step_%d_rts_l2_dist'
    byt = pre_dobj['byt']

    batch_size = len(batch_tup[0])
    steps = [i + 1 for i in range(PGD_SAVE_PERIOD - 1, config.s2_iters, PGD_SAVE_PERIOD)]
    for step in steps:
        rts_at_step = pre_dobj[rts_step_template % step]
        logits_at_step = pre_dobj[logits_step_template % step]
        pred_at_step = np.argmax(logits_at_step, -1)
        dobj[pred_step_template % step] = pred_at_step
        succeed_at_step = np.logical_and(pred_at_step == byt, byt != batch_tup[1]).astype(np.int64)
        dobj[succeed_step_template % step] = succeed_at_step
        diff_at_step = (rts_at_step - rts_benign).reshape((batch_size, -1))
        rts_l1_dist_at_step = np.linalg.norm(diff_at_step, 1, axis=-1)
        rts_l2_dist_at_step = np.linalg.norm(diff_at_step, 2, axis=-1)
        dobj[rts_l1_dist_step_template % step] = rts_l1_dist_at_step
        dobj[rts_l2_dist_step_template % step] = rts_l2_dist_at_step
        print('step', succeed_at_step.mean())

    return dobj


def attack(config):
    if config.model == 'resnet50':
        rts_model = RTSResnet50(RTS_RESNET50_CKPT_DIR, config.device == 'gpu')
    if config.model == 'densenet169':
        rts_model = RTSDensenet169(RTS_DENSENET169_CKPT_DIR, config.device == 'gpu')
        freeze_model(rts_model.blackbox_model)
    freeze_model(rts_model.saliency)

    data_arx = np.load(config.data_path)
    rts_benign_arx = np.load(config.rts_benign_path)
    img_x, img_y, img_yt = data_arx['img_x'], data_arx['img_y'].copy(), data_arx['img_yt']
    rts_benign = rts_benign_arx['saliency_benign_y'].copy()

    n, batch_size = len(img_x), config.batch_size
    num_batches = (n + batch_size - 1) // batch_size
    save_dobjs = []
    for i in range(num_batches):
        si = i * batch_size
        ei = min(si + batch_size, n)
        batch_tup = (img_x[si:ei], img_y[si:ei], img_yt[si:ei])
        bm0 = rts_benign[si:ei]
        dobj = attack_batch(config, rts_model, batch_tup, bm0)
        dobj.update(analyze_batch(config, rts_model,batch_tup, bm0, dobj))
        save_dobjs.append(dobj)

    keys = list(save_dobjs[0].keys())
    save_dobj = {}
    for key in keys:
        save_dobj[key] = np.concatenate([i[key] for i in save_dobjs], axis=0)
    np.savez(config.save_path, **save_dobj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('rts_benign_path')
    parser.add_argument('model', choices=['resnet50', 'densenet169'])
    parser.add_argument('save_path')
    parser.add_argument('-d', '--device', dest='device', choices=['cpu', 'gpu'])
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=40)
    parser.add_argument('-e', '--epsilon', dest='epsilon', type=float, default=0.031)
    parser.add_argument('--s1-iters', dest='s1_iters', type=int, default=400)
    parser.add_argument('--s1-lr', dest='s1_lr', type=float, default=1./255)
    parser.add_argument('--s2-iters', dest='s2_iters', type=int, default=1200)
    parser.add_argument('--s2-lr', dest='s2_lr', type=float, default=1./255)
    parser.add_argument('-c', dest='c', type=float, default=1.)
    config = parser.parse_args()

    if config.device is None:
        if torch.cuda.is_available():
            config.device = 'gpu'
        else:
            config.device = 'cpu'
    print('Please check the configuration', config)
    attack(config)
