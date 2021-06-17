#!/usr/bin/env python
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
import torch.autograd as autograd

from expr_attacks.utils import freeze_model
from ia_utils.data_utils import imagenet_normalize
from ia_utils.resnet_srelu import resnet50 as resnet50_soft
from rev2.gs.generate_gs import generate_gs_per_batches, imagenet_resize_postfn


def load_model():
    model = resnet50(True)
    model_ref = resnet50_soft(True)
    model.to('cuda')
    model.train(False)
    model_ref.to('cuda')
    model_ref.train(False)
    freeze_model(model)
    freeze_model(model_ref)
    return (model, imagenet_normalize), (model_ref, imagenet_normalize)


def attack_batch(config, model_tup, ref_model_tup, bx, by, bm):
    n = len(bx)
    model, pre_fn = model_tup
    ref_model_tup = (ref_model_tup[0], lambda x: x)

    best_dist = np.full((n, ), np.inf, dtype=np.float32)
    best_adv = bx.cpu().numpy()
    best_adv_gs = np.zeros((n, 56, 56), dtype=np.float32)
    best_adv_conf = np.zeros((n,), dtype=np.float32)

    bx0 = bx.clone()
    bx = bx.clone().requires_grad_()

    for i in range(300):
        bx_p = pre_fn(bx)
        logit = model(bx_p)
        adv_loss = F.nll_loss(F.log_softmax(logit), by, reduction='sum')
        final_grad = autograd.grad([adv_loss], [bx])[0]

        bx.data = bx.data - 1. / 255 * final_grad.sign()
        r = bx.data - bx0
        r.clamp_(-0.031, 0.031)
        bx.data = bx0 + r
        del final_grad

    bx_adv_start = bx.detach().clone()
    bx = bx_adv_start.clone().requires_grad_()

    for c, num_step in zip(config.cs, config.steps):
        for i in range(num_step):
            bx_p = pre_fn(bx)
            logit = model(bx_p)
            adv_loss = F.nll_loss(F.log_softmax(logit), by, reduction='sum')
            adv_gs = generate_gs_per_batches(ref_model_tup, bx_p, by, post_fn=imagenet_resize_postfn,
                                             keep_grad=True)

            if i % 10 == 0:
                with torch.no_grad():
                    prob = F.softmax(logit).gather(1, by.view(n, -1)).view(n)
                prob = prob.cpu().numpy()
                now_gs = generate_gs_per_batches(model_tup, bx, by, post_fn=imagenet_resize_postfn)
                diff = now_gs.detach() - bm
                now_dist = (diff * diff).view(n, -1).sum(1).cpu().numpy()
                mask = np.logical_and(prob > 0.8, now_dist < best_dist)
                indices_np = np.nonzero(mask)[0]
                indices = torch.tensor(indices_np, device='cuda')
                best_dist[indices_np] = now_dist[indices_np]
                best_adv[indices_np] = bx.detach()[indices].cpu().numpy()
                best_adv_gs[indices_np] = now_gs[indices].cpu().numpy()
                best_adv_conf[indices_np] = prob[indices_np]

            diff = adv_gs - bm
            int_loss = (diff * diff).view(n, -1).sum()
            loss = adv_loss + c * int_loss
            final_grad = autograd.grad([loss], [bx])[0]

            bx.data = bx.data - 1./255 * final_grad.sign()
            r = bx.data - bx0
            r.clamp_(-0.031, 0.031)
            bx.data = bx0 + r
            bx.data.clamp_(0, 1)

            if i % 10 == 0:
                succeed_indices = np.nonzero(best_dist < np.inf)[0]

                print('c', c, 'step', i,
                      'succeed:', len(succeed_indices), 'conf:', np.mean(best_adv_conf[succeed_indices]),
                      'dist', np.mean(best_dist[succeed_indices]))

            del final_grad, loss, int_loss
    return dict(best_dist=best_dist, best_adv=best_adv, best_adv_gs=best_adv_gs, best_adv_conf=best_adv_conf)


def attack(config):
    model_tup, ref_model_tup = load_model()

    data_arx = np.load(config.data_path)
    img_x, img_yt = data_arx['att_imgs'], data_arx['att_yts']
    gs_target = data_arx['att_tgs']
    gs_benign = data_arx['att_bgs']

    if config.begin != -1 and config.end != -1:
        beg, end = config.begin, config.end
        img_x, img_yt, bs_benign, gs_target = img_x[beg:end], img_yt[beg:end], gs_benign[beg:end], gs_target[beg:end]

    n, batch_size = len(img_x), config.batch_size
    num_batches = (n + batch_size - 1) // batch_size
    save_dobjs = []
    print(n)
    for i in range(num_batches):
        si = i * batch_size
        ei = min(si + batch_size, n)
        bx_np, byt_np, bm0_np = img_x[si:ei], img_yt[si:ei], gs_target[si:ei]
        bx, byt, bm0 = [torch.tensor(t, device='cuda') for t in (bx_np, byt_np, bm0_np)]
        dobj = attack_batch(config, model_tup, ref_model_tup, bx, byt, bm0)
        save_dobjs.append(dobj)

    keys = list(save_dobjs[0].keys())
    save_dobj = {}
    for key in keys:
        save_dobj[key] = np.concatenate([i[key] for i in save_dobjs], axis=0)
    save_dobj['img_x'] = img_x
    save_dobj['img_yt'] = img_yt
    save_dobj['att_tgs'] = gs_target
    save_dobj['att_bgs'] = gs_benign
    np.savez(config.save_path, **save_dobj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('save_path')
    parser.add_argument('-b', '--batch-size', type=int, dest='batch_size', default=10)

    parser.add_argument('-s', '--steps', dest='steps', type=int, nargs='+', default=[301, 201, 101, 101])
    parser.add_argument('-c', dest='cs', type=float, nargs='+', default=[0.0001, 0.001, 0.005, 0.02])
    parser.add_argument('--begin', dest='begin', type=int, default=-1)
    parser.add_argument('--end', dest='end', type=int, default=-1)
    config = parser.parse_args()

    print('Please check the configuration', config)
    attack(config)
