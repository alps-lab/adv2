#!/usr/bin/env python
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd as autograd
from torchvision.models.resnet import resnet50
from expr_attacks.mask_attack_utils import Adam
from expr_attacks.mask_model import MASKV2, mask_iter_v2
from expr_attacks.mask_generator import generate_mask_per_batch_v2
from expr_attacks.commons import PGD_SAVE_PERIOD
from expr_attacks.utils import freeze_model
from ia_utils.data_utils import imagenet_normalize


def attack_batch(config, model_tup, mask_model, batch_tup, m0):
    cuda = config.device == 'gpu'
    bx_np, by_np = batch_tup
    m0_np = m0
    batch_size = len(bx_np)
    bx, by, m0 = torch.tensor(bx_np), torch.tensor(by_np), torch.tensor(m0)
    if cuda:
        bx, by, m0 = bx.cuda(), by.cuda(), m0.cuda()
    bx_adv = bx.clone().detach()
    bx_adv.requires_grad = True
    model, pre_fn = model_tup[:2]
    dobj = {}
    m = torch.empty_like(m0).fill_(0.5)
    m.requires_grad = True

    s1_lr = config.s1_lr
    s2_beta = config.s2_beta
    eps = config.epsilon

    for i in range(config.s1_iters):
        logits = model(pre_fn(bx_adv))
        pgd_loss = torch.sum(logits.scatter(1, by[:, None], -100000).max(1)[0] - logits.gather(1, by[:, None])[:, 0])
        grad = autograd.grad([pgd_loss], [bx_adv])[0]
        with torch.no_grad():
            bx_adv.data = bx_adv.data - s1_lr * grad.sign()
            diff = bx_adv - bx
            diff.clamp_(-eps, eps)
            bx_adv.data = bx + diff
            bx_adv.data.clamp_(0, 1)

    r = (bx_adv - bx).detach()
    r.requires_grad = True
    optimizer = Adam(0.1)

    mean_his = [torch.zeros_like(m0)]
    variance_his = [torch.zeros_like(m0)]
    no_backprop_steps = 4
    backprop_steps = 4
    adam_step = 0
    attack_mask_config = dict(lr=0.1, l1_lambda=config.mask_l1_lambda, tv_lambda=config.mask_tv_lambda,
                              noise_std=0, n_iters=config.mask_iter,
                              verbose=False)

    best_l2_dists = np.full(len(bx), 1e10, dtype=np.float32)
    best_adv_x = np.zeros_like(bx_np)
    best_adv_mask = np.zeros_like(m0_np)

    for i in range(config.s2_iters):

        if i % PGD_SAVE_PERIOD == PGD_SAVE_PERIOD - 1:
            bx_adv = (bx + r).detach_()
            bx_adv_disc_np = np.uint8(255 * bx_adv.cpu().numpy())
            bx_adv_disc = torch.tensor(np.float32(bx_adv_disc_np / 255.))
            if cuda:
                bx_adv_disc = bx_adv_disc.cuda()
            mask_now_disc = generate_mask_per_batch_v2(attack_mask_config, mask_model, model_tup,
                                               batch_tup=(bx_adv_disc, by), cuda=cuda).detach_()
            with torch.no_grad():
                logits_disc = model(pre_fn(bx_adv_disc))
            succeed = (logits_disc.argmax(1) == by).long().cpu().numpy()

            with torch.no_grad():
                diff = m0 - mask_now_disc
                diff_norm = diff.view(batch_size, -1).norm(2, 1).cpu().numpy()
            update_flag = np.logical_and(diff_norm < best_l2_dists, succeed.astype(np.bool))
            bx_adv_disc_np = bx_adv_disc.cpu().numpy()
            mask_now_disc_np = mask_now_disc.cpu().numpy()
            best_adv_x[update_flag] = bx_adv_disc_np[update_flag]
            best_l2_dists[update_flag] = diff_norm[update_flag]
            best_adv_mask[update_flag] = mask_now_disc_np[update_flag]

            print('step', i, 'l2 dist_disc', np.asscalar(diff_norm.mean()), 'succeed_disc', np.asscalar(succeed.astype(np.float32).mean()))
            m.data = mask_now_disc.data
            mean_his = [0.5 * mean_his[0]] # mean_his[0].detach()]
            variance_his = [0.25 * variance_his[0]]
            adam_step = 100

        logits = model(pre_fn(bx + r))
        pgd_loss = torch.sum(
            logits.scatter(1, by[:, None], -100000).max(1)[0] - logits.gather(1, by[:, None])[:, 0])
        pgd_grad = autograd.grad([pgd_loss], [r])[0]

        with torch.no_grad():
            pgd_grad_norm = torch.norm(pgd_grad.view(batch_size, -1), 2, 1, keepdim=True).view(batch_size, 1, 1, 1)
            r.data -= s2_beta * pgd_grad / pgd_grad_norm
            r.data.clamp_(-eps, eps)
            r.data = torch.clamp(bx + r, 0, 1) - bx

        for j in range(no_backprop_steps):
            int_loss = mask_iter_v2(mask_model, model, pre_fn,
                                    F.interpolate(bx + r.detach(), (228, 228), mode='bilinear'),
                                    by, m, noise_std=0,
                                    l1_lambda=config.mask_l1_lambda, tv_lambda=config.mask_tv_lambda)[0]
            int_grad = autograd.grad([int_loss], [m])[0]
            objs = optimizer.__call__(adam_step + 1, [m], [int_grad], mean_his, variance_his, False)
            m = objs[0][0]
            m = torch.clamp(m, 0, 1)
            mean_his = objs[-2]
            variance_his = objs[-1]
            m = m.detach()
            m.requires_grad = True
            adam_step += 1

        diffs = []
        for j in range(backprop_steps):
            int_loss = mask_iter_v2(mask_model, model, pre_fn,
                                    F.interpolate(bx + r, (228, 228), mode='bilinear'),
                                    by, m, noise_std=0,
                                    l1_lambda=config.mask_l1_lambda, tv_lambda=config.mask_tv_lambda)[0]
            int_grad = autograd.grad([int_loss], [m], create_graph=True)[0]
            objs = optimizer.__call__(adam_step + 1, [m], [int_grad], mean_his, variance_his, True)
            m = objs[0][0]
            m = torch.clamp(m, 0, 1)
            mean_his = objs[-2]
            variance_his = objs[-1]
            diff = m - m0
            diffs.append((diff * diff).sum())
            m = m.detach()
            m.requires_grad = True
            adam_step += 1

        int_final_loss = torch.stack(diffs).mean()
        final_grad = autograd.grad([int_final_loss], [r])[0]
        assert np.asscalar(torch.isnan(final_grad).max()) != 1
        with torch.no_grad():
            final_grad_norm = torch.norm(final_grad.view(batch_size, -1), 2, 1, keepdim=True).view(batch_size, 1, 1, 1)
            final_grad_normed = final_grad / final_grad_norm

        left, right = torch.zeros_like(by).float().fill_(0), torch.zeros_like(by).float().fill_(
            0.08 if i < 900 else 0.04)
        bs_step = 0
        r.detach_()
        best_step_size = left.clone()
        while bs_step < 10:
            middle = (left + right) * 0.5
            r_test = r - middle[:, None, None, None] * final_grad_normed
            r_test.clamp_(-eps, eps)
            new_x = (bx + r_test).clamp_(0, 1)
            with torch.no_grad():
                logits = model(pre_fn(new_x))
                pgd_loss = torch.sum(logits.scatter(1, by[:, None], -100000).max(1)[0]
                                     - logits.gather(1, by[:, None])[:, 0])
                mask_ = pgd_loss < -5.0
                left.masked_scatter_(mask_, middle)
                right.masked_scatter_(mask_ ^ 1, middle)
                best_step_size.masked_scatter_(mask_, torch.max(middle, best_step_size))
            bs_step += 1

        r = r - best_step_size[:, None, None, None] * final_grad_normed
        r.clamp_(-eps, eps)
        r = torch.clamp(bx + r, 0, 1) - bx
        r.requires_grad = True
        mean_his = [mean_his[0].detach()]
        variance_his = [variance_his[0].detach()]

    dobj['adv_x'] = best_adv_x
    dobj['adv_mask'] = best_adv_mask
    dobj['adv_succeed'] = (best_l2_dists < 1e6).astype(np.int64)
    dobj['tmask'] = m0_np
    return dobj


def attack(config):
    mask_model = MASKV2(config.device == 'gpu')
    model = resnet50(pretrained=True)
    model.train(False)
    if config.device == 'gpu':
        model.cuda()
    freeze_model(model)
    model_tup = (model, imagenet_normalize, (224, 224))

    data_arx = np.load(config.data_path)
    img_x, img_y, img_yt = data_arx['att_imgs'], data_arx['att_ys'], data_arx['att_yts']
    mask_target = data_arx['att_tmasks']
    mask_benign = data_arx['att_bmasks']

    n, batch_size = len(img_x), config.batch_size
    num_batches = (n + batch_size - 1) // batch_size
    save_dobjs = []
    print(n)
    for i in range(num_batches):
        si = i * batch_size
        ei = min(si + batch_size, n)
        bx, byt, bm0 = img_x[si:ei], img_yt[si:ei], mask_target[si:ei]
        dobj = attack_batch(config, model_tup, mask_model, (bx, byt), bm0)
        dobj['bmask'] = mask_benign[si:ei]
        save_dobjs.append(dobj)

    keys = list(save_dobjs[0].keys())
    save_dobj = {}
    for key in keys:
        save_dobj[key] = np.concatenate([i[key] for i in save_dobjs], axis=0)
    save_dobj.update(dict(img_x=img_x, img_y=img_y, img_yt=img_yt))
    np.savez(config.save_path, **save_dobj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('save_path')
    parser.add_argument('-d', '--device', dest='device', choices=['cpu', 'gpu'])
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=10)
    parser.add_argument('-e', '--epsilon', dest='epsilon', type=float, default=0.031)
    parser.add_argument('--s1-iters', dest='s1_iters', type=int, default=400)
    parser.add_argument('--s1-lr', dest='s1_lr', type=float, default=1. / 255)
    parser.add_argument('--s2-iters', dest='s2_iters', type=int, default=1000)
    parser.add_argument('--s2-beta', dest='s2_beta', type=float, default=0.05)
    parser.add_argument('--mask-iter', dest='mask_iter', type=int, default=300)
    parser.add_argument('--mask-noise-std', dest='mask_noise_std', type=float, default=0)
    parser.add_argument('--mask-tv-lambda', dest='mask_tv_lambda', type=float, default=1e-2)
    parser.add_argument('--mask-l1-lambda', dest='mask_l1_lambda', type=float, default=1e-4)
    config = parser.parse_args()

    if config.device is None:
        if torch.cuda.is_available():
            config.device = 'gpu'
        else:
            config.device = 'cpu'
    print('Please check the configuration', config)
    attack(config)
