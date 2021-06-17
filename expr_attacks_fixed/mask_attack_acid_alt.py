#!/usr/bin/env python
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd as autograd
from torch.optim import Adam
from torchvision.models.resnet import resnet50
from torchvision.models.vgg import vgg16
from torchvision.models.densenet import densenet169

from ia_utils.data_utils import imagenet_normalize
from expr_attacks.utils import freeze_model
from expr_attacks.mask_model import MASK, mask_iter
from expr_attacks.mask_generator import generate_mask_per_batch
from expr_attacks.commons import PGD_SAVE_PERIOD


def load_model(config):
    pre_fn = imagenet_normalize
    if config.model == 'vgg16':
        model = vgg16(pretrained=True)
        shape = (224, 224)
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

    s1_lr = config.s1_lr
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
        pgd_loss = F.nll_loss(F.log_softmax(logits, dim=-1), by, reduction='sum')
        grad = autograd.grad([pgd_loss], [bx_adv])[0]
        with torch.no_grad():
            bx_adv.data = bx_adv.data - s1_lr * grad.sign()
            diff = bx_adv - bx
            diff.clamp_(-eps, eps)
            bx_adv.data = bx + diff
            bx_adv.data.clamp_(0, 1)

    r = (bx_adv - bx).detach()
    r.requires_grad = True

    optimizer = Adam(params=[r], lr=1e-2, amsgrad=True)
    c_start, c_end = np.log(0.001), np.log(5)
    c_now, c_inc = c_start, (c_end - c_start) / config.s2_iters
    p_start, p_end = -np.log(0.5), -np.log(0.99)
    p_now, p_inc = p_start, (p_end - p_start) / config.s2_iters

    attack_mask_config = dict(lr=0.1, l1_lambda=1e-2, tv_lambda=1e-4, noise_std=0, n_iters=500,
                              verbose=False)
    for i in range(config.s2_iters):
        c_now = c_now + c_inc
        p_now = p_now + p_inc
        if i % PGD_SAVE_PERIOD == PGD_SAVE_PERIOD - 1 or i == 0:
            bx_adv = (bx + r).detach_()
            mask_now = generate_mask_per_batch(attack_mask_config, mask_model, model_tup,
                                               batch_tup=(bx_adv, by), cuda=cuda, m_init=m0)
            mask_now.detach_()
            bx_adv_disc_np = np.uint8(255 * bx_adv.cpu().numpy())
            bx_adv_disc = torch.tensor(np.float32(bx_adv_disc_np / 255.))
            if cuda:
                bx_adv_disc = bx_adv_disc.cuda()
            random_start = (torch.rand_like(m0) - 0.5) * 0.6
            random_start = m0.detach() + random_start
            random_start.clamp_(0, 1)
            mask_now_disc = generate_mask_per_batch(attack_mask_config, mask_model, model_tup,
                                               batch_tup=(bx_adv_disc, by), cuda=cuda, m_init=random_start).detach_()
            with torch.no_grad():
                logits = model(pre_fn(bx + r))
                logits_disc = model(pre_fn(bx_adv_disc))
            dobj[adv_step_template % (i + 1)] = bx_adv.cpu().numpy()
            dobj[adv_disc_step_template % (i + 1)] = bx_adv_disc_np
            # dobj[mask_m_step_template % (i + 1)] = m.detach().cpu().numpy()
            dobj[mask_mnow_step_template % (i + 1)] = mask_now.cpu().numpy()
            dobj[mask_disc_mnow_step_template % (i + 1)] = mask_now_disc.cpu().numpy()
            dobj[logits_step_template % (i + 1)] = logits.detach().cpu().numpy()
            dobj[logits_disc_step_template % (i + 1)] = logits_disc.detach().cpu().numpy()

            with torch.no_grad():
                diff = mask_now_disc - m0
                diff = np.asscalar((diff * diff).sum())
            print('step', i, 'l2 dist now', diff, 'succeed', np.asscalar((logits_disc.argmax(1) == by).float().mean()),
                  'succeed_disc', np.asscalar((logits_disc.argmax(1) == by).float().mean()))

        adv = bx + r
        mask_loss = mask_iter(mask_model, model, pre_fn, adv, by, None, m0, noise_std=0.)[0]
        logits = model(pre_fn(adv))
        adv_loss = F.nll_loss(F.log_softmax(logits, -1), by, reduction='none')
        total_loss = mask_loss + float(np.exp(c_now)) * F.relu(
            torch.pow(adv_loss, 2) - float(p_now * p_now)).sum()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            r.data.clamp_(-eps, eps)
            r.data = torch.clamp(bx + r, 0, 1) - bx
        print('step', i, mask_loss.item() / batch_size, adv_loss.mean().item())

    return dobj


def attack(config):
    data_arx = np.load(config.data_path)
    mask_benign_arx = np.load(config.mask_benign_path)
    img_x, img_y, img_yt = (data_arx['img_x'].copy(), data_arx['img_y'].copy(),
                            data_arx['img_yt'].copy())
    mask_benign = mask_benign_arx['mask_benign_y'].copy()
    mask_model = MASK(config.device == 'gpu')
    model_tup = load_model(config)

    n, batch_size = len(img_x), config.batch_size
    num_batches = (n + batch_size - 1) // batch_size
    save_dobjs = []
    for i in range(num_batches):
        si = i * batch_size
        ei = min(si + batch_size, n)
        bx, byt, bm0 = img_x[si:ei], img_yt[si:ei], mask_benign[si:ei]
        dobj = attack_batch(config, model_tup, mask_model, (bx, byt), bm0)
        dobj.update(analyze_batch(config, mask_model, model_tup, (bx, byt), bm0, dobj))
        save_dobjs.append(dobj)
        print('done batch: %d' % i)

    keys = list(save_dobjs[0].keys())
    save_dobj = {}
    for key in keys:
        save_dobj[key] = np.concatenate([i[key] for i in save_dobjs], axis=0)
    save_dobj['img_x'] = img_x
    save_dobj['img_y'] = img_y
    save_dobj['img_yt'] = img_yt
    save_dobj['target_mask'] = mask_benign
    np.savez(config.save_path, **save_dobj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('mask_benign_path')
    parser.add_argument('model', choices=['resnet50', 'densenet169'])
    parser.add_argument('save_path')
    parser.add_argument('-d', '--device', dest='device', choices=['cpu', 'gpu'])
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=7)
    parser.add_argument('-e', '--epsilon', dest='epsilon', type=float, default=0.031)
    parser.add_argument('--s1-iters', dest='s1_iters', type=int, default=100)
    parser.add_argument('--s1-lr', dest='s1_lr', type=float, default=1. / 255)
    parser.add_argument('--s2-iters', dest='s2_iters', type=int, default=400)
    parser.add_argument('--s2-beta', dest='s2_beta', type=float, default=1. / 255)
    parser.add_argument('--mask-iter', dest='mask_iter', type=int, default=500)
    parser.add_argument('--mask-noise-std', dest='mask_noise_std', type=float, default=0)
    parser.add_argument('--mask-tv-lambda', dest='mask_tv_lambda', type=float, default=1e-4)
    parser.add_argument('--mask-l1-lambda', dest='mask_l1_lambda', type=float, default=1e-2)
    parser.add_argument('-c', dest='c', type=float, default=5.)
    config = parser.parse_args()

    if config.device is None:
        if torch.cuda.is_available():
            config.device = 'gpu'
        else:
            config.device = 'cpu'
    print('Please check the configuration', config)
    attack(config)
