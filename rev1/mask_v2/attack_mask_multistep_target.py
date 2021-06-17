#!/usr/bin/env python
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd as autograd
from torchvision.models.resnet import resnet50
from torchvision.models.densenet import densenet169

from ia_utils.data_utils import imagenet_normalize
from expr_attacks.utils import freeze_model
from expr_attacks.mask_attack_utils import Adam
from expr_attacks.mask_model import MASKV2, mask_iter_v2
from expr_attacks.mask_generator import generate_mask_per_batch_v2
from expr_attacks.commons import PGD_SAVE_PERIOD


def load_model(config):
    pre_fn = imagenet_normalize
    if config.model == 'resnet50':
        model = resnet50(pretrained=True)
        shape = (224, 224)
    if config.model == 'densenet169':
        model = densenet169(pretrained=True)
        shape = (224, 224)
    freeze_model(model)
    model.train(False)
    if config.device == 'gpu':
        model.cuda()
    return model, pre_fn, shape


def analyze_batch(config, mask_model, model_tup, batch_tup, mask_benign, pre_dobj):
    by = batch_tup[1]
    logits_step_template = 'pgd_s2_step_%d_adv_logits'
    logits_disc_step_template = 'pgd_s2_step_%d_adv_logits_disc'
    preds_step_template = 'pgd_s2_step_%d_preds'
    preds_disc_step_template = 'pgd_s2_step_%d_preds_disc'
    succeed_step_tamplate = 'pgd_s2_step_%d_succeed'
    succeed_disc_step_tamplate = 'pgd_s2_step_%d_succeed_disc'

    dobj = {}
    for i in range(PGD_SAVE_PERIOD - 1, config.s2_iters, PGD_SAVE_PERIOD):
        step = i + 1
        logits = pre_dobj[logits_step_template % step]
        preds = np.argmax(logits, -1)
        succeed = (preds == by).astype(np.int64)
        dobj[preds_step_template % step] = preds
        dobj[succeed_step_tamplate % step] = succeed
        logits_disc = pre_dobj[logits_disc_step_template % step]
        preds_disc = np.argmax(logits_disc, -1)
        succeed_disc = (preds_disc == by).astype(np.int64)
        dobj[preds_disc_step_template % step] = preds_disc
        dobj[succeed_disc_step_tamplate % step] = succeed_disc
    return dobj


def attack_batch(config, model_tup, mask_model, batch_tup, m0):
    cuda = config.device == 'gpu'
    bx_np, by_np = batch_tup
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
    adv_step_template = 'pgd_s2_step_%d_adv_x'
    mask_m_step_template = 'pgd_s2_step_%d_adv_mask_m'  # current values of `m`
    mask_mnow_step_template = 'pgd_s2_step_%d_adv_mask_mnow'  # current optimal MASK
    logits_step_template = 'pgd_s2_step_%d_adv_logits'
    adv_disc_step_template = 'pgd_s2_step_%d_adv_x_disc'
    mask_disc_mnow_step_template = 'pgd_s2_step_%d_adv_mask_mnow_disc'  # current optimal MASK
    logits_disc_step_template = 'pgd_s2_step_%d_adv_logits_disc'

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
                logits = model(pre_fn(bx + r))
                logits_disc = model(pre_fn(bx_adv_disc))
            dobj[adv_step_template % (i + 1)] = bx_adv.cpu().numpy()
            dobj[adv_disc_step_template % (i + 1)] = bx_adv_disc_np
            dobj[mask_m_step_template % (i + 1)] = m.detach().cpu().numpy()
            dobj[mask_disc_mnow_step_template % (i + 1)] = mask_now_disc.cpu().numpy()
            dobj[logits_step_template % (i + 1)] = logits.detach().cpu().numpy()
            dobj[logits_disc_step_template % (i + 1)] = logits_disc.detach().cpu().numpy()

            with torch.no_grad():
                diff = m0 - mask_now_disc
                diff = np.asscalar((diff * diff).sum())
            print('step', i, 'l2 dist now', diff, 'succeed', np.asscalar((logits_disc.argmax(1) == by).float().mean()),
                  'succeed_disc', np.asscalar((logits_disc.argmax(1) == by).float().mean()))
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
        # print(i, np.asscalar(diffs[-1]))
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
                mask_ = pgd_loss < -10.0
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

    return dobj


def attack(config):
    mask_target_arx = np.load(config.mask_target_path)
    img_x, img_y, img_yt = (mask_target_arx['img_x'].copy(), mask_target_arx['img_y'].copy(),
                            mask_target_arx['img_yt'].copy())
    mask_target = mask_target_arx['target_mask'].copy()
    mask_model = MASKV2(config.device == 'gpu')
    model_tup = load_model(config)

    n, batch_size = len(img_x), config.batch_size
    num_batches = (n + batch_size - 1) // batch_size
    save_dobjs = []
    for i in range(num_batches):
        si = i * batch_size
        ei = min(si + batch_size, n)
        bx, byt, bm0 = img_x[si:ei], img_yt[si:ei], mask_target[si:ei]
        dobj = attack_batch(config, model_tup, mask_model, (bx, byt), bm0)
        dobj.update(analyze_batch(config, mask_model, model_tup, (bx, byt), bm0, dobj))
        save_dobjs.append(dobj)
        print('done batch: %d' % i)

    keys = list(save_dobjs[0].keys())
    save_dobj = {}
    for key in keys:
        save_dobj[key] = np.concatenate([i[key] for i in save_dobjs], axis=0)
    save_dobj.update(dict(img_x=img_x, img_y=img_y, img_yt=img_yt, target_mask=mask_target))
    np.savez(config.save_path, **save_dobj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mask_target_path')
    parser.add_argument('-model', choices=['resnet50', 'densenet169'], default='resnet50')
    parser.add_argument('save_path')
    parser.add_argument('-d', '--device', dest='device', choices=['cpu', 'gpu'])
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=12)
    parser.add_argument('-e', '--epsilon', dest='epsilon', type=float, default=0.031)
    parser.add_argument('--s1-iters', dest='s1_iters', type=int, default=400)
    parser.add_argument('--s1-lr', dest='s1_lr', type=float, default=1. / 255)
    parser.add_argument('--s2-iters', dest='s2_iters', type=int, default=1000)
    parser.add_argument('--s2-beta', dest='s2_beta', type=float, default=0.05)
    parser.add_argument('--mask-iter', dest='mask_iter', type=int, default=300)
    parser.add_argument('--mask-noise-std', dest='mask_noise_std', type=float, default=0)
    parser.add_argument('--mask-tv-lambda', dest='mask_tv_lambda', type=float, default=1e-2)
    parser.add_argument('--mask-l1-lambda', dest='mask_l1_lambda', type=float, default=1e-4)
    parser.add_argument('-c', dest='c', type=float, default=5.)
    config = parser.parse_args()

    if config.device is None:
        if torch.cuda.is_available():
            config.device = 'gpu'
        else:
            config.device = 'cpu'
    print('Please check the configuration', config)
    attack(config)
