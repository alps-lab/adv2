#!/usr/bin/env python
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from expr_attacks.cam_model_def import cam_resnet50, cam_densenet169
from expr_attacks.cam_model import cam_forward
from expr_attacks.commons import PGD_SAVE_PERIOD
from expr_attacks.utils import freeze_model
from rev1.stadv_acid.stadv_def import StadvFlow, StadvFlowLoss, StadvTVLoss


def load_model(config):
    if config.model == 'resnet50':
        model_tup, forward_tup = cam_resnet50()
    if config.model == 'densenet169':
        model_tup, forward_tup = cam_densenet169()
    model = model_tup[0]
    freeze_model(model)
    model.train(False)
    if config.device == 'gpu':
        model.cuda()
    return model_tup, forward_tup


def attack_batch(config, model_tup, forward_tup, batch_tup, cam_benign):
    model_tup, forward_tup = load_model(config)
    model, pre_fn = model_tup[:2]
    device = 'cuda' if config.device == 'gpu' else 'cpu'
    # cuda = config.device == 'gpu'
    bx_np, by_np = batch_tup
    batch_size = len(bx_np)
    bx, by, m0 = (torch.tensor(bx_np, device=device), torch.tensor(by_np, device=device),
              torch.tensor(cam_benign, device=device))
    m0_flatten = m0.view(batch_size, -1)

    adv_step_template = 's2_step_%d_stadv_x'
    cam_step_template = 's2_step_%d_stadv_cam'
    cam_n_step_template = 's2_step_%d_stadv_cam_n'
    logits_step_template = 's2_step_%d_stadv_logits'
    flow_step_template = 's2_step_%d_stadv_flow'
    flow_loss_step_template = 's2_step_%d_stadv_flow_loss'
    flow_tvloss_step_template = 's2_step_%d_stadv_flow_tvloss'
    dobj = {}

    images = bx
    flows = 0.2 * (torch.rand(batch_size, 2, images.size(2), images.size(3), device=device) - 0.5)
    flows.requires_grad_(True)

    tau = config.tau
    flow_obj = StadvFlow()
    flow_loss_obj = StadvFlowLoss()
    flow_tvloss_obj = StadvTVLoss()
    optimizer = Adam([flows], lr=0.01, amsgrad=True)

    for i in range(config.s1_iters):
        adv_images = flow_obj(images, flows)
        logits = model(pre_fn(adv_images))
        adv_loss = F.nll_loss(F.log_softmax(logits, dim=-1), by, reduction='none')
        flow_loss = flow_loss_obj(flows)
        total_loss = adv_loss + tau * flow_loss

        optimizer.zero_grad()
        total_loss.sum().backward()
        optimizer.step()
        if i % 50 == 0 or i == config.s1_iters - 1:
            with torch.no_grad():
                flow_loss = flow_tvloss_obj(flows)
                preds = logits.argmax(1)
                succeed = (preds == by).float().mean().item()
            print('s1-step: %d, average adv loss: %.4f, average flow loss: %.4f, succeed: %.2f' %
                  (i, adv_loss.mean().item(), flow_loss.mean().item(), succeed))

    optimizer = Adam([flows], lr=0.01, amsgrad=True)
    c_begin, c_final = config.c, config.c * 2
    c_inc = (c_final - c_begin) / config.s2_iters
    c_now = config.c
    for i in range(config.s2_iters):
        c_now += c_inc
        adv_images = flow_obj(images, flows)
        flow_loss = flow_loss_obj(flows)

        logits, cam = cam_forward(model_tup, forward_tup, adv_images, by)
        adv_loss = F.nll_loss(F.log_softmax(logits, dim=-1), by, reduction='none')
        cam_flatten = cam.view(batch_size, -1)
        cam_flatten = cam_flatten - cam_flatten.min(1, True)[0]
        cam_flatten = cam_flatten / cam_flatten.max(1, True)[0]
        diff = cam_flatten - m0_flatten
        loss_cam = (diff * diff).mean(1)
        total_loss = 2 * adv_loss + tau * flow_loss + c_now * loss_cam

        # record examples
        if i % PGD_SAVE_PERIOD == PGD_SAVE_PERIOD - 1:
            cam_flatten_n = cam_flatten.view(*cam.size()).detach()
            dobj[adv_step_template % (i + 1)] = adv_images.detach().cpu().numpy()
            dobj[cam_step_template % (i + 1)] = cam.detach().cpu().numpy()
            dobj[cam_n_step_template % (i + 1)] = cam_flatten_n.cpu().numpy()
            dobj[logits_step_template % (i + 1)] = logits.detach().cpu().numpy()
            dobj[flow_step_template % (i + 1)] = flows.detach().cpu().numpy()

            with torch.no_grad():
                flow_loss = flow_loss_obj(flows, 0.)
                flow_tvloss = flow_tvloss_obj(flows)
            dobj[flow_loss_step_template % (i + 1)] = flow_loss.cpu().numpy()
            dobj[flow_tvloss_step_template % (i + 1)] = flow_tvloss.cpu().numpy()

        optimizer.zero_grad()
        total_loss.sum().backward()
        optimizer.step()

        # print message
        if i % 100 == 0:
            with torch.no_grad():
                pred = torch.argmax(logits, 1)
                loss_cam_mu = loss_cam.mean().item()
                loss_adv_mu = adv_loss.mean().item()
                flow_loss = flow_tvloss_obj(flows).mean().item()
                num_succeed = np.asscalar(torch.sum(by == pred))
                adv_loss = loss_adv_mu
                loss_cam = loss_cam_mu
            print('s2-step: %d, loss flow: %.3f, loss adv: %.2f, loss cam: %.5f, succeed: %d' %
                  (i, flow_loss, adv_loss, loss_cam, num_succeed))
    return dobj


def analyze_batch(config, model_tup, forward_tup, batch_tup, cam_benign, pre_dobj):
    dobj = {}
    cam_n_step_template = 's2_step_%d_stadv_cam_n'
    logits_step_template = 's2_step_%d_stadv_logits'
    succeed_step_template = 's2_step_%d_stadv_succeed'
    pred_step_template = 's2_step_%d_pred'
    cam_l1_dist_step_template = 's2_step_%d_cam_l1_dist'
    cam_l2_dist_step_template = 's2_step_%d_cam_l2_dist'

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
    cam_benign_y = cam_benign_arx['cam_benign_yt_n'].copy()

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
    save_dobj['cam_benign_y_n'] = cam_benign_y
    save_dobj["img_x"] = img_x
    np.savez(config.save_path, **save_dobj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('cam_benign_path')
    parser.add_argument('-model', choices=['resnet50', 'densenet169'], default='resnet50')
    parser.add_argument('save_path')
    parser.add_argument('-d', '--device', dest='device', choices=['cpu', 'gpu'])
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=50)
    parser.add_argument('--s1-iters', dest='s1_iters', type=int, default=200)
    parser.add_argument('--s2-iters', dest='s2_iters', type=int, default=600)
    parser.add_argument('-c', dest='c', type=float, default=5.)
    parser.add_argument('--tau', dest='tau', type=float, default=0.0005)
    config = parser.parse_args()

    if config.device is None:
        if torch.cuda.is_available():
            config.device = 'gpu'
        else:
            config.device = 'cpu'
    print('Please check the configuration', config)
    attack(config)
