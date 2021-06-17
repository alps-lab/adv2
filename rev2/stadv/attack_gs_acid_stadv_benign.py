#!/usr/bin/env python
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from ia_utils.model_utils import resnet50, densenet169, freeze_model
from ia_utils.data_utils import imagenet_normalize
from ia_utils.resnet_srelu import resnet50 as resnet50_soft
from ia_utils.densenet_srelu import densenet169 as densenet169_soft
from rev2.gs.generate_gs import generate_gs_per_batches, imagenet_resize_postfn
from rev1.stadv_acid.stadv_def import StadvFlow, StadvFlowLoss, StadvTVLoss


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


def attack_batch(config, model_tup, ref_model_tup, img_x, img_y, gs_benign):
    bx_np, by_np = img_x, img_y
    n = len(bx_np)
    model, pre_fn = model_tup[:2]
    ref_model_tup = (ref_model_tup[0], lambda x: x)

    device = 'cuda' if config.device == 'cuda' else 'cpu'
    bx, by, m0 = (torch.tensor(bx_np, device=device), torch.tensor(by_np, device=device),
              torch.tensor(gs_benign, device=device))
    images = bx
    flows = 0.2 * (torch.rand(n, 2, images.size(2), images.size(3), device=device) - 0.5)
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
    del optimizer

    cs = [0.003, 0.01, 0.05, 0.2]
    steps = [301, 201, 101, 101]

    best_dist = np.full((n, ), np.inf, dtype=np.float32)
    best_adv = bx.cpu().numpy()
    best_adv_gs = np.zeros((n, 56, 56), dtype=np.float32)
    best_adv_conf = np.zeros((n,), dtype=np.float32)
    best_adv_flow = np.zeros((n, 2, 224, 224), dtype=np.float32)

    for cur_c, cur_tot_step in zip(cs, steps):
        optimizer = Adam([flows], lr=0.01, amsgrad=True)
        for i in range(cur_tot_step):
            adv_images = flow_obj(images, flows)
            adv_p = pre_fn(adv_images)
            logit = model(adv_p)

            adv_loss = F.nll_loss(F.log_softmax(logit), by, reduction='sum')
            adv_gs = generate_gs_per_batches(ref_model_tup, adv_p, by, post_fn=imagenet_resize_postfn,
                                             keep_grad=True)

            if i % 10 == 0:
                with torch.no_grad():
                    prob = F.softmax(logit).gather(1, by.view(n, -1)).view(n)
                prob = prob.cpu().numpy()
                now_gs = generate_gs_per_batches(model_tup, adv_images, by, post_fn=imagenet_resize_postfn)
                diff = now_gs.detach() - m0
                now_dist = (diff * diff).view(n, -1).sum(1).cpu().numpy()
                mask = np.logical_and(prob > 0.75, now_dist < best_dist)
                indices_np = np.nonzero(mask)[0]
                indices = torch.tensor(indices_np, device=config.device)
                best_dist[indices_np] = now_dist[indices_np]
                best_adv[indices_np] = adv_images.detach()[indices].cpu().numpy()
                best_adv_gs[indices_np] = now_gs[indices].cpu().numpy()
                best_adv_conf[indices_np] = prob[indices_np]
                best_adv_flow[indices_np] = flows.detach()[indices].cpu().numpy()

            diff = adv_gs - m0
            int_loss = (diff * diff).view(n, -1).sum()
            flow_loss = flow_loss_obj(flows).sum()
            loss = adv_loss + cur_c * int_loss + tau * flow_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                succeed_indices = np.nonzero(best_dist < np.inf)[0]
                with torch.no_grad():
                    flow_loss = flow_tvloss_obj(flows)

                print('c:', cur_c, 'step:', i,
                      'succeed:', len(succeed_indices), 'conf:', np.mean(best_adv_conf[succeed_indices]),
                      'dist:', np.mean(best_dist[succeed_indices]),
                      'avg flow loss:', flow_loss.mean())

            del loss, int_loss, flow_loss
        del optimizer
    return dict(best_dist=best_dist, best_adv=best_adv, best_adv_gs=best_adv_gs, best_adv_conf=best_adv_conf,
                best_adv_flow=best_adv_flow, succeed=(best_dist < np.inf).astype(np.int64))


def attack(config):
    model_tup, ref_model_tup = load_model(config)
    data_arx = np.load(config.data_path)
    gs_benign_arx = np.load(config.gs_benign_path)
    img_x, img_yt = data_arx['img_x'].copy(), data_arx['img_yt'].copy()
    gs_benign_y = gs_benign_arx['benign_gs'].copy()

    if config.begin != -1 and config.end != -1:
        beg, end = config.begin, config.end
        img_x, img_yt, gs_benign_y = img_x[beg:end], img_yt[beg:end], gs_benign_y[beg:end]

    n, batch_size = len(img_x), config.batch_size
    num_batches = (n + batch_size - 1) // batch_size
    save_dobjs = []
    for i in range(num_batches):
        si = i * batch_size
        ei = min(si + batch_size, n)
        bx, by = (img_x[si:ei], img_yt[si:ei])
        dobj = attack_batch(config, model_tup, ref_model_tup, bx, by, gs_benign_y[si:ei])
        save_dobjs.append(dobj)
    keys = list(save_dobjs[0].keys())
    save_dobj = {}
    for key in keys:
        save_dobj[key] = np.concatenate([i[key] for i in save_dobjs], axis=0)
    save_dobj['benign_gs'] = gs_benign_y
    save_dobj["img_x"] = img_x
    np.savez(config.save_path, **save_dobj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('gs_benign_path')
    parser.add_argument('-model', choices=['resnet50', 'densenet169'], default='resnet50')
    parser.add_argument('save_path')
    parser.add_argument('-d', '--device', dest='device', choices=['cpu', 'cuda'])
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=10)
    parser.add_argument('--s1-iters', dest='s1_iters', type=int, default=450)
    parser.add_argument('--tau', dest='tau', type=float, default=0.0005)
    parser.add_argument('--begin', dest='begin', type=int, default=-1)
    parser.add_argument('--end', dest='end', type=int, default=-1)
    config = parser.parse_args()

    if config.device is None:
        if torch.cuda.is_available():
            config.device = 'cuda'
        else:
            config.device = 'cpu'
    print('Please check the configuration', config)
    attack(config)
