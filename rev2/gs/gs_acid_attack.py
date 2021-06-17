#!/usr/bin/env python
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd as autograd

from ia_utils.model_utils import resnet50, densenet169, freeze_model
from ia_utils.data_utils import imagenet_normalize
from ia_utils.resnet_srelu import resnet50 as resnet50_soft
from ia_utils.densenet_srelu import densenet169 as densenet169_soft
from rev2.gs.generate_gs import generate_gs_per_batches, imagenet_resize_postfn


def load_model(config):
    if config.model == 'resnet':
        model = resnet50(True)
        model_ref = resnet50_soft(True)
    else:
        model = densenet169(True)
        model_ref = densenet169_soft(True)
    model.to(config.device)
    model.train(False)
    model_ref.to(config.device)
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
                indices = torch.tensor(indices_np, device=config.device)
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
    return dict(best_dist=best_dist, best_adv=best_adv, best_adv_gs=best_adv_gs)


def analyze_batch(config, model_tup, bx, by, bm, result):
    n = len(bx)
    succeed = (result['best_dist'] < np.inf).astype(np.int64)
    diff = (bm - result['best_adv_gs']).reshape((n, -1))
    return dict(succeed=succeed, l2_dist=np.linalg.norm(diff, 2, axis=1),
                l1_dist=np.linalg.norm(diff, 1, axis=1))


def main(config):
    device = config.device
    model_tup, ref_model_tup = load_model(config)
    dobj = np.load(config.data_path)
    img_x, img_y, img_m = dobj['img_x'], dobj['img_yt'], dobj['benign_gs']
    if config.begin != -1 and config.end != -1:
        beg, end = config.begin, config.end
        img_x, img_y, img_m = img_x[beg:end], img_y[beg:end], img_m[beg:end]
    n, batch_size = len(img_x), config.batch_size
    n_batches = (n + batch_size - 1) // batch_size
    results = []
    for i in range(n_batches):
        si = i * batch_size
        ei = min(n, si + batch_size)
        bx_np, by_np, bm_np = img_x[si:ei], img_y[si:ei], img_m[si:ei]
        bx, by, bm = [torch.tensor(arr, device=device) for arr in (bx_np, by_np, bm_np)]
        result = attack_batch(config, model_tup, ref_model_tup, bx, by, bm)
        result.update(analyze_batch(config, model_tup, bx_np, by_np, bm_np, result))
        results.append(result)

    keys = list(results[0].keys())
    save_dobj = {}
    for key in keys:
        save_dobj[key] = np.concatenate([i[key] for i in results], axis=0)
    save_dobj['img_x'] = img_x
    save_dobj['img_y'] = dobj['img_y']
    save_dobj['img_yt'] = img_y
    save_dobj['benign_gs'] = img_m
    np.savez(config.save_path, **save_dobj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['resnet', 'densenet'])
    parser.add_argument('data_path')
    parser.add_argument('save_path')
    parser.add_argument('-b', '--batch-size', type=int, dest='batch_size', default=10)
    parser.add_argument('-d', '--device', dest='device', choices=['cpu', 'gpu'])
    parser.add_argument('-s', '--steps', dest='steps', type=int, nargs='+', default=[301, 201, 101, 101])
    parser.add_argument('-c', dest='cs', type=float, nargs='+', default=[0.001, 0.004, 0.01, 0.05])
    parser.add_argument('--begin', dest='begin', type=int, default=-1)
    parser.add_argument('--end', dest='end', type=int, default=-1)

    config = parser.parse_args()
    if config.device is None:
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if config.device == 'gpu':
        config.device = 'cuda'
    print('configuration:', config)
    main(config)
