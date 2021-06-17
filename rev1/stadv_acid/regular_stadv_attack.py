#!/usr/bin/env python
import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.models.resnet import resnet50
from torchvision.models.vgg import vgg16
from torchvision.models.densenet import densenet169

from ia_utils.data_utils import imagenet_normalize
from expr_attacks.utils import freeze_model
from rev1.stadv_acid.stadv_def import StadvFlow, StadvFlowLoss, StadvTVLoss


def load_model(config):
    pre_fn = imagenet_normalize
    if config.model == 'resnet50':
        model = resnet50(pretrained=True)
        shape = (224, 224)
    if config.model == 'densenet169':
        model = densenet169(pretrained=True)
        shape = (224, 224)
    model.train(False)
    if config.device == 'gpu':
        model.cuda()
    return model, pre_fn, shape


def vanilla_attack_batch(config, model_tup, batch_tup):
    device = 'cuda' if config.device == 'gpu' else 'cpu'
    model, pre_fn, shape = model_tup
    bx_np, byt_np = batch_tup

    dobj = {}
    batch_size = len(bx_np)
    bx, byt = torch.tensor(bx_np, device=device), torch.tensor(byt_np, device=device)

    images = bx
    flows = 0.2 * (torch.rand(batch_size, 2, images.size(2), images.size(3), device=device) - 0.5)
    flows.requires_grad_(True)

    with torch.no_grad():
        logits_reg = model(pre_fn(bx)).cpu().numpy()
    dobj['logits_benign'] = logits_reg

    flow_obj = StadvFlow()
    flow_loss_obj = StadvFlowLoss()
    flow_tvloss_obj = StadvTVLoss()
    optimizer = Adam([flows], lr=0.01, amsgrad=True)

    for j in range(config.max_iters):
        adv_images = flow_obj(images, flows)
        logits = model(pre_fn(adv_images))
        adv_loss = F.nll_loss(F.log_softmax(logits, dim=-1), byt, reduction='none')
        flow_loss = flow_loss_obj(flows)
        total_loss = adv_loss + config.c * flow_loss

        optimizer.zero_grad()
        total_loss.sum().backward()
        optimizer.step()
        if j % 50 == 0 or j == config.max_iters - 1:
            with torch.no_grad():
                flow_loss = flow_tvloss_obj(flows)
                preds = logits.argmax(1)
                succeed = (preds == byt).float().mean().item()
            print('step: %d, average adv loss: %.4f, average flow loss: %.4f, succeed: %.2f' %
                  (j, adv_loss.mean().item(), flow_loss.mean().item(), succeed))

    dobj['adv_flow'] = flows.detach().cpu().numpy()
    with torch.no_grad():
        adv_images = flow_obj(images, flows)
        logits = model(pre_fn(adv_images))
        preds = logits.argmax(1)
        succeed = (preds == byt).long()
        flow_loss = flow_loss_obj(flows, 0.)
        flow_tvloss = flow_tvloss_obj(flows)

    dobj['stadv_x'] = adv_images.cpu().numpy()
    dobj['stadv_logits'] = logits.cpu().numpy()
    dobj['succeed'] = succeed.cpu().numpy()
    dobj['stadv_flow_loss'] = flow_loss.cpu().numpy()
    dobj['stadv_flow_tvloss'] = flow_tvloss.cpu().numpy()

    return dobj


def analyze_batch(config, model_tup, batch_tup, dobj):
    bx_np, byt_np = batch_tup
    new_dobj = {}
    logits_adv = dobj['stadv_logits']
    y_adv = np.argmax(logits_adv, axis=-1)
    new_dobj['stadv_succeed'] = (y_adv == byt_np).astype(np.int64)
    print('successful rate: %.4f' % (y_adv == byt_np).astype(np.int64).mean())
    return new_dobj


def attack_batch(config, model_tup, batch_tup):
    dobj = {}
    dobj.update(vanilla_attack_batch(config, model_tup, batch_tup))
    dobj.update(analyze_batch(config, model_tup, batch_tup, dobj))
    return dobj


def attack(config, model_tup, images_tup):
    img_x, img_yt = images_tup
    n = len(img_x)
    num_batches = (n + config.batch_size - 1) // config.batch_size
    batch_dobjs = []

    for i in range(num_batches):
        start_index = i * config.batch_size
        end_index = min(n, start_index + config.batch_size)
        bx_np, byt_np = (img_x[start_index:end_index], img_yt[start_index:end_index])
        robj = attack_batch(config, model_tup, (bx_np, byt_np))
        batch_dobjs.append(robj)
    keys = batch_dobjs[0].keys()
    robj = {}
    for key in keys:
        robj[key] = np.concatenate([dobj[key] for dobj in batch_dobjs], axis=0)
    robj["benign_x"] = img_x
    np.savez(config.save_path, **robj)


def main(config):
    model, pre_fn, shape = load_model(config)
    freeze_model(model)
    npobj = np.load(config.data_path)
    img_x, img_y, img_yt, img_path = (npobj['img_x'].copy(), npobj['img_y'].copy(),
                                      npobj['img_yt'].copy(),
                                      npobj['img_path'].copy())
    attack(config, (model, pre_fn, shape), (img_x, img_yt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='.npz file of images to attack')
    parser.add_argument('model', choices=['resnet50', 'densenet169'])
    parser.add_argument('save_path')
    parser.add_argument('-b', '--batch-size', type=int, dest='batch_size', default=40)
    parser.add_argument('-d', '--device', choices=['cpu', 'gpu'], dest='device')
    parser.add_argument('-a', '--alpha', type=float, dest='alpha', default=1/255.)
    parser.add_argument('-c', type=float, dest='c', default=0.0005)
    parser.add_argument('--max-iters', type=int, dest='max_iters', default=300)

    config = parser.parse_args()
    if config.device is None:
        if torch.cuda.is_available():
            config.device = 'gpu'
        else:
            config.device = 'cpu'
    print('Please check the configuration', config)
    main(config)
