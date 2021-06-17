#!/usr/bin/env python
import argparse

from progressbar import progressbar
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from rev2.cifar10.model_utils import resnet50, CIFAR10_RESNET50_CKPT_PATH
from rev2.cifar10.data_utils import cifar10_normalize

from expr_attacks.commons import PGD_MAXITERS, PGD_SAVE_PERIOD
from expr_attacks.utils import freeze_model


def load_model(config):
    pre_fn = cifar10_normalize
    model = resnet50()
    nn.DataParallel(model).load_state_dict(torch.load(CIFAR10_RESNET50_CKPT_PATH,
                                           lambda storage, location: storage)['net'])
    shape = (32, 32)
    model.train(False)
    if config.device == 'cuda':
        model.cuda()
    return model, pre_fn, shape


def vanilla_attack_batch(config, model_tup, batch_tup):
    model, pre_fn, shape = model_tup
    bx_np, byt_np = batch_tup

    dobj = {}
    batch_size = len(bx_np)
    bx, byt = torch.tensor(bx_np), torch.tensor(byt_np)
    best_loss = torch.zeros(batch_size).fill_(np.inf)
    if config.device == 'cuda':
        bx, byt = bx.cuda(), byt.cuda()
        best_loss = best_loss.cuda()
    bx_adv = bx.clone().detach()
    bx_adv.requires_grad = True
    bx_adv_found = bx.clone().detach()

    with torch.no_grad():
        logits_reg = model(pre_fn(bx)).cpu().numpy()
    dobj['logits_benign'] = logits_reg

    for j in range(config.max_iters):
        logits = model(pre_fn(bx_adv))
        loss = F.nll_loss(F.log_softmax(logits, dim=-1), byt, reduction='none')
        if j % 250 == 0:
            print('step: %d, average loss: %.4f' % (j, np.asscalar(loss.mean())))
        pgd_grad = autograd.grad([loss.sum()], [bx_adv])[0]
        mask = torch.nonzero(loss < best_loss)[:, 0]
        bx_adv_found.data[mask] = bx_adv.data[mask]
        with torch.no_grad():
            bx_adv_new = bx_adv - config.alpha * pgd_grad.sign()
            diff = bx_adv_new - bx
            diff.clamp_(-config.epsilon, config.epsilon)
            bx_adv.data = bx.data + diff
            bx_adv.data.clamp_(0, 1)

        if j % PGD_SAVE_PERIOD == PGD_SAVE_PERIOD - 1:
            with torch.no_grad():
                logits_adv = model(pre_fn(bx_adv_found))
            logits_adv = logits_adv.cpu().numpy()
            dobj['pgd_step_%d_logits_adv' % (j + 1)] = logits_adv
            dobj['pgd_step_%d_adv_x' % (j + 1)] = bx_adv_found.detach().cpu().numpy()
    return dobj


def analyze_batch(config, model_tup, batch_tup, dobj):
    bx_np, byt_np = batch_tup
    new_dobj = {}
    steps = [i + 1 for i in range(PGD_SAVE_PERIOD - 1, config.max_iters, PGD_SAVE_PERIOD)]
    for step in steps:
        logits_adv = dobj['pgd_step_%d_logits_adv' % step]
        y_adv = np.argmax(logits_adv, axis=-1)
        new_dobj['pgd_step_%d_succeed' % step] = (y_adv == byt_np).astype(np.int64)
        print('step: %d, successful rate: %.4f' % (step, (y_adv == byt_np).astype(np.int64).mean()))
    return new_dobj


def attack_batch(config, model_tup, batch_tup):
    model, pre_fn, shape = model_tup
    bx_np, byt_np = batch_tup

    dobj = {}
    dobj.update(vanilla_attack_batch(config, model_tup, batch_tup))
    dobj.update(analyze_batch(config, model_tup, batch_tup, dobj))
    return dobj


def attack(config, model_tup, images_tup):
    img_x, img_y, img_yt = images_tup
    n = len(img_x)
    num_batches = (n + config.batch_size - 1) // config.batch_size
    batch_dobjs = []

    for i in progressbar(range(num_batches)):
        start_index = i * config.batch_size
        end_index = min(n, start_index + config.batch_size)
        bx_np, byt_np = (img_x[start_index:end_index], img_yt[start_index:end_index])
        robj = attack_batch(config, model_tup, (bx_np, byt_np))
        batch_dobjs.append(robj)
    keys = batch_dobjs[0].keys()
    robj = {}
    for key in keys:
        robj[key] = np.concatenate([dobj[key] for dobj in batch_dobjs], axis=0)
    robj["img_x"] = img_x
    robj['img_y'] = img_y
    robj['img_yt'] = img_yt
    np.savez(config.save_path, **robj)


def main(config):
    model, pre_fn, shape = load_model(config)
    freeze_model(model)
    npobj = np.load(config.data_path)
    img_x, img_y, img_yt = (npobj['img_x'].copy(), npobj['img_y'].copy(),
                            npobj['img_yt'].copy())
    attack(config, (model, pre_fn, shape), (img_x, img_y, img_yt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='.npz file of images to attack')
    parser.add_argument('save_path')
    parser.add_argument('-b', '--batch-size', type=int, dest='batch_size', default=40)
    parser.add_argument('-d', '--device', choices=['cpu', 'cuda'], dest='device')
    parser.add_argument('-a', '--alpha', type=float, dest='alpha', default=1/255.)
    parser.add_argument('-e', '--epsilon', type=float, dest='epsilon', default=0.031)
    parser.add_argument('--max-iters', type=int, dest='max_iters', default=PGD_MAXITERS)

    config = parser.parse_args()
    if config.device is None:
        if torch.cuda.is_available():
            config.device = 'cuda'
        else:
            config.device = 'cpu'
    print('Please check the configuration', config)
    main(config)
