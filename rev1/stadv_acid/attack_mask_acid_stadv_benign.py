#!/usr/bin/env python
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd as autograd
from torch.optim import Adam as TorchAdam
from torchvision.models.resnet import resnet50
from torchvision.models.densenet import densenet169

from ia_utils.data_utils import imagenet_normalize
from expr_attacks.utils import freeze_model
from expr_attacks.mask_attack_utils import Adam
from expr_attacks.mask_model import MASKV2, mask_iter_v2
from expr_attacks.mask_generator import generate_mask_per_batch_v2
from expr_attacks.commons import PGD_SAVE_PERIOD
from rev1.stadv_acid.stadv_def import StadvFlow, StadvFlowLoss, StadvTVLoss


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
    logits_disc_step_template = 'pgd_s2_step_%d_adv_logits_disc'
    preds_disc_step_template = 'pgd_s2_step_%d_preds_disc'
    succeed_disc_step_tamplate = 'pgd_s2_step_%d_succeed_disc'

    dobj = {}
    for i in range(PGD_SAVE_PERIOD - 1, config.s2_iters, PGD_SAVE_PERIOD):
        step = i + 1
        logits_disc = pre_dobj[logits_disc_step_template % step]
        preds_disc = np.argmax(logits_disc, -1)
        succeed_disc = (preds_disc == by).astype(np.int64)
        dobj[preds_disc_step_template % step] = preds_disc
        dobj[succeed_disc_step_tamplate % step] = succeed_disc
    return dobj


def attack_batch(config, model_tup, mask_model, batch_tup, m0):
    device = 'cuda' if config.device == 'gpu' else 'cpu'
    bx_np, by_np = batch_tup
    batch_size = len(bx_np)
    bx, by, m0 = (torch.tensor(bx_np, device=device), torch.tensor(by_np, device=device),
              torch.tensor(m0, device=device))
    model, pre_fn = model_tup[:2]
    dobj = {}
    m = torch.empty_like(m0).fill_(0.5)
    m.requires_grad = True

    mask_m_step_template = 'pgd_s2_step_%d_adv_mask_m'  # current values of `m`
    adv_disc_step_template = 'pgd_s2_step_%d_adv_x_disc'
    mask_disc_mnow_step_template = 'pgd_s2_step_%d_adv_mask_mnow_disc'  # current optimal MASK
    logits_disc_step_template = 'pgd_s2_step_%d_adv_logits_disc'
    flow_step_template = 's2_step_%d_stadv_flow'
    flow_loss_step_template = 's2_step_%d_stadv_flow_loss'
    flow_tvloss_step_template = 's2_step_%d_stadv_flow_tvloss'

    images = bx
    flows = 0.2 * (torch.rand(batch_size, 2, images.size(2), images.size(3), device=device) - 0.5)
    flows.requires_grad_(True)

    tau = config.tau
    flow_obj = StadvFlow()
    flow_loss_obj = StadvFlowLoss()
    flow_tvloss_obj = StadvTVLoss()
    optimizer = TorchAdam([flows], lr=0.01, amsgrad=True)

    for i in range(config.s1_iters):
        adv_images = flow_obj(images, flows)
        logits = model(pre_fn(adv_images))
        adv_loss = logits.scatter(1, by[:, None], -100000).max(1)[0] - logits.gather(1, by[:, None])[:, 0]
        adv_loss = torch.max(adv_loss, torch.full_like(adv_loss, -5.))
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

    optimizer = Adam(0.01)
    torch_optimizer = TorchAdam([flows], lr=0.005, amsgrad=True)

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
            with torch.no_grad():
                adv_images = flow_obj(images, flows)
            mask_now_disc = generate_mask_per_batch_v2(attack_mask_config, mask_model, model_tup,
                                                       batch_tup=(adv_images_, by),
                                                       cuda=config.device == 'gpu').detach_()
            with torch.no_grad():
                logits_disc = model(pre_fn(adv_images_))
            dobj[adv_disc_step_template % (i + 1)] = adv_images_.cpu().numpy()
            dobj[mask_m_step_template % (i + 1)] = m.detach().cpu().numpy()
            dobj[mask_disc_mnow_step_template % (i + 1)] = mask_now_disc.cpu().numpy()
            dobj[logits_disc_step_template % (i + 1)] = logits_disc.detach().cpu().numpy()
            dobj[flow_step_template % (i + 1)] = flows.detach().cpu().numpy()

            with torch.no_grad():
                flow_loss = flow_loss_obj(flows, 0.)
                flow_tvloss = flow_tvloss_obj(flows)
            dobj[flow_loss_step_template % (i + 1)] = flow_loss.cpu().numpy()
            dobj[flow_tvloss_step_template % (i + 1)] = flow_tvloss.cpu().numpy()

            with torch.no_grad():
                diff = m0 - mask_now_disc
                diff = np.asscalar((diff * diff).sum())
                flow_tvloss = flow_tvloss_obj(flows).mean().item()
                adv_loss = logits.scatter(1, by[:, None], -100000).max(1)[0] - logits.gather(1, by[:, None])[:, 0]
                adv_loss = torch.max(adv_loss, torch.full_like(adv_loss, -5.))

            print('step', i, 'l2 dist now', diff, 'succeed', np.asscalar((logits_disc.argmax(1) == by).float().mean()),
                  'succeed_disc', np.asscalar((logits_disc.argmax(1) == by).float().mean()),
                  'flow tvloss', flow_tvloss, 'adv loss', adv_loss.mean().item())
            m.data = mask_now_disc.data
            mean_his = [0.5 * mean_his[0]] # mean_his[0].detach()]
            variance_his = [0.25 * variance_his[0]]
            adam_step = 100

        for j in range(no_backprop_steps):
            adv_images_ = adv_images.detach()
            int_loss = mask_iter_v2(mask_model, model, pre_fn,
                                    F.interpolate(adv_images_, (228, 228), mode='bilinear'),
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
            adv_images = flow_obj(images, flows)
            int_loss = mask_iter_v2(mask_model, model, pre_fn,
                                    F.interpolate(adv_images, (228, 228), mode='bilinear'),
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
        adv_images = flow_obj(images, flows)
        logits = model(pre_fn(adv_images))
        adv_loss = logits.scatter(1, by[:, None], -100000).max(1)[0] - logits.gather(1, by[:, None])[:, 0]
        adv_loss = torch.max(adv_loss, torch.full_like(adv_loss, -5.))
        total_loss = 500 * int_final_loss + adv_loss.sum() + 0.004 * flow_loss_obj(flows).sum()

        torch_optimizer.zero_grad()
        total_loss.backward()
        torch_optimizer.step()

    return dobj


def attack(config):
    mask_benign_arx = np.load(config.mask_benign_path)
    img_x, img_y, img_yt = (mask_benign_arx['img_x'].copy(), mask_benign_arx['img_y'].copy(),
                            mask_benign_arx['img_yt'].copy())
    mask_benign = mask_benign_arx['mask_benign_y'].copy()
    mask_model = MASKV2(config.device == 'gpu')
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
    save_dobj.update(dict(img_x=img_x, img_y=img_y, img_yt=img_yt, mask_benign_y=mask_benign))
    np.savez(config.save_path, **save_dobj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('mask_benign_path')
    parser.add_argument('-model', choices=['resnet50', 'densenet169'], default='resnet50')
    parser.add_argument('save_path')
    parser.add_argument('-d', '--device', dest='device', choices=['cpu', 'gpu'])
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=10)
    parser.add_argument('--s1-iters', dest='s1_iters', type=int, default=400)
    parser.add_argument('--s2-iters', dest='s2_iters', type=int, default=1000)
    parser.add_argument('--mask-iter', dest='mask_iter', type=int, default=300)
    parser.add_argument('--mask-noise-std', dest='mask_noise_std', type=float, default=0)
    parser.add_argument('--mask-tv-lambda', dest='mask_tv_lambda', type=float, default=1e-2)
    parser.add_argument('--mask-l1-lambda', dest='mask_l1_lambda', type=float, default=1e-4)
    parser.add_argument('--tau', dest='tau', type=float, default=0.0005)
    parser.add_argument('-c', dest='c', type=float, default=5.)
    config = parser.parse_args()

    if config.device is None:
        if torch.cuda.is_available():
            config.device = 'gpu'
        else:
            config.device = 'cpu'
    print('Please check the configuration', config)
    attack(config)
