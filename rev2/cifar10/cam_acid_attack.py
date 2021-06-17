#!/usr/bin/env python
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from rev2.cifar10.cam_model_def import cam_resnet50
from expr_attacks.cam_model import cam_forward
from expr_attacks.commons import PGD_SAVE_PERIOD
from expr_attacks.utils import freeze_model


def load_model(config):
    model_tup, forward_tup = cam_resnet50()
    model = model_tup[0]
    freeze_model(model)
    model.train(False)
    if config.device == 'gpu':
        model.cuda()
    return model_tup, forward_tup


def attack_batch(config, model_tup, forward_tup, batch_tup, cam_benign):
    model_tup, forward_tup = load_model(config)
    model, pre_fn = model_tup[:2]
    cuda = config.device == 'gpu'
    bx_np, by_np = batch_tup
    batch_size = len(bx_np)
    bx, by = torch.tensor(bx_np), torch.tensor(by_np)
    m0 = torch.tensor(cam_benign)
    if cuda:
        bx, by, m0 = bx.cuda(), by.cuda(), m0.cuda()
    m0_flatten = m0.view(batch_size, -1)
    bx_adv = bx.clone().detach().requires_grad_()

    s1_lr = config.s1_lr
    s2_lr = config.s2_lr
    eps = config.epsilon
    adv_step_template = 'pgd_s2_step_%d_adv_x'
    cam_step_template = 'pgd_s2_step_%d_adv_cam'
    cam_n_step_template = 'pgd_s2_step_%d_adv_cam_n'
    logits_step_template = 'pgd_s2_step_%d_adv_logits'
    dobj = {}

    for i in range(config.s1_iters):
        logits = model(pre_fn(bx_adv))
        loss = F.nll_loss(F.log_softmax(logits), by, reduction='sum')
        loss_grad = autograd.grad([loss], [bx_adv])[0]

        # first record message
        if i % 100 == 0:
            with torch.no_grad():
                loss_adv_mu = np.asscalar(loss) / batch_size
                pred = torch.argmax(logits, 1)
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
        logits, cam = cam_forward(model_tup, forward_tup, bx_adv, by)
        loss_adv = F.nll_loss(F.log_softmax(logits), by, reduction='sum')
        cam_flatten = cam.view(batch_size, -1)
        cam_flatten = cam_flatten - cam_flatten.min(1, True)[0]
        cam_flatten = cam_flatten / cam_flatten.max(1, True)[0]
        diff = cam_flatten - m0_flatten
        loss_cam = torch.sum((diff * diff).mean(1))
        loss = torch.add(loss_adv, c_now, loss_cam)
        loss_grad = autograd.grad([loss], [bx_adv])[0]

        # record examples
        if i % PGD_SAVE_PERIOD == PGD_SAVE_PERIOD - 1:
            cam_flatten_n = cam_flatten.view(*cam.size()).detach()
            dobj[adv_step_template % (i + 1)] = bx_adv.detach().cpu().numpy()
            dobj[cam_step_template % (i + 1)] = cam.detach().cpu().numpy()
            dobj[cam_n_step_template % (i + 1)] = cam_flatten_n.cpu().numpy()
            dobj[logits_step_template % (i + 1)] = logits.detach().cpu().numpy()

        # print message
        if i % 100 == 0:
            with torch.no_grad():
                pred = torch.argmax(logits, 1)
                loss_cam_mu = np.asscalar(loss_cam) / batch_size
                loss_adv_mu = np.asscalar(loss_adv) / batch_size
                num_succeed = np.asscalar(torch.sum(by == pred))
                loss_adv = loss_adv_mu
                loss_cam = loss_cam_mu
            print('s2-step: %d, loss adv: %.2f, loss cam: %.5f, succeed: %d' % (i, loss_adv, loss_cam, num_succeed))

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


def analyze_batch(config, model_tup, forward_tup, batch_tup, cam_benign, pre_dobj):
    dobj = {}
    cam_n_step_template = 'pgd_s2_step_%d_adv_cam_n'
    logits_step_template = 'pgd_s2_step_%d_adv_logits'
    succeed_step_template = 'pgd_s2_step_%d_adv_succeed'
    pred_step_template = 'pgd_s2_step_%d_pred'
    cam_l1_dist_step_template = 'pgd_s2_step_%d_cam_l1_dist'
    cam_l2_dist_step_template = 'pgd_s2_step_%d_cam_l2_dist'

    batch_size = len(batch_tup[0])
    steps = [i + 1 for i in range(PGD_SAVE_PERIOD - 1, config.s2_iters, PGD_SAVE_PERIOD)]
    for step in steps:
        cam_at_step = pre_dobj[cam_n_step_template % step]
        logits_at_step = pre_dobj[logits_step_template % step]
        pred_at_step = np.argmax(logits_at_step, -1)
        dobj[pred_step_template % step] = pred_at_step
        succeed_at_step = (pred_at_step == batch_tup[1]).astype(np.int64)
        dobj[succeed_step_template % step] = succeed_at_step
        diff_at_step = (cam_at_step - cam_benign).reshape((batch_size, -1))
        cam_l1_dist_at_step = np.linalg.norm(diff_at_step, 1, axis=-1)
        cam_l2_dist_at_step = np.linalg.norm(diff_at_step, 2, axis=-1)
        dobj[cam_l1_dist_step_template % step] = cam_l1_dist_at_step
        dobj[cam_l2_dist_step_template % step] = cam_l2_dist_at_step

    return dobj


def attack(config):
    model_tup, forward_tup = cam_resnet50()
    data_arx = np.load(config.data_path)
    cam_benign_arx = np.load(config.cam_benign_path)
    img_x, img_yt = data_arx['img_x'].copy(), data_arx['img_yt'].copy()
    cam_benign_y = cam_benign_arx['cam_benign_y_n'].copy()

    n, batch_size = len(img_x), config.batch_size
    num_batches = (n + batch_size - 1) // batch_size
    save_dobjs = []
    for i in range(num_batches):
        si = i * batch_size
        ei = min(si + batch_size, n)
        batch_tup = (img_x[si:ei], img_yt[si:ei])
        dobj = attack_batch(config, model_tup, forward_tup, batch_tup, cam_benign_y[si:ei])
        dobj.update(analyze_batch(config, model_tup, forward_tup, batch_tup, cam_benign_y[si:ei], dobj))
        save_dobjs.append(dobj)
    keys = list(save_dobjs[0].keys())
    save_dobj = {}

    for key in keys:
        save_dobj[key] = np.concatenate([i[key] for i in save_dobjs], axis=0)
    save_dobj['img_x'] = img_x
    save_dobj['img_yt'] = img_yt
    save_dobj['cam_benign_y_n'] = cam_benign_y

    np.savez(config.save_path, **save_dobj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('cam_benign_path')
    parser.add_argument('save_path')
    parser.add_argument('-d', '--device', dest='device', choices=['cpu', 'gpu'])
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=50)
    parser.add_argument('-e', '--epsilon', dest='epsilon', type=float, default=0.031)
    parser.add_argument('--s1-iters', dest='s1_iters', type=int, default=400)
    parser.add_argument('--s1-lr', dest='s1_lr', type=float, default=1./255)
    parser.add_argument('--s2-iters', dest='s2_iters', type=int, default=1200)
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
