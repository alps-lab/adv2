import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from expr_attacks.mask_model import MASK, MASKV2, mask_iter, mask_iter_v2


def get_default_mask_config():
    return dict(lr=0.1, l1_lambda=1e-2, tv_lambda=1e-4, noise_std=0, n_iters=400,
                batch_size=40, verbose=False)


def generate_mask_per_batch(mask_config, mask_model, model_tup, batch_tup, cuda, m_init=None):
    bx, by = batch_tup
    batch_size = len(bx)
    if not isinstance(bx, torch.Tensor):
        bx = torch.tensor(bx)
    if not isinstance(by, torch.Tensor):
        by = torch.tensor(by)
    model, pre_fn, shape = model_tup
    if m_init is None:
        m_init = torch.zeros(batch_size, 1, 28, 28).fill_(0.5)
    else:
        m_init = m_init.detach()
    if cuda:
        bx, by = bx.cuda(), by.cuda()
        m_init = m_init.cuda()
    m_init.requires_grad = True
    optimizer = Adam([m_init], lr=mask_config['lr'])
    bx_blurred = mask_model.gaussian_blur(bx)
    for i in range(mask_config['n_iters']):
        tot_loss = mask_iter(mask_model, model, pre_fn, bx, by, None,
                             m_init, mask_config['l1_lambda'], mask_config['tv_lambda'],
                             noise_std=mask_config['noise_std'], x_blurred=bx_blurred)[0]
        if mask_config['verbose'] and i % 50 == 0:
            print(i, np.asscalar(tot_loss) / batch_size)
        optimizer.zero_grad()
        tot_loss.backward()
        optimizer.step()
        m_init.data.clamp_(0, 1)
    return m_init


def generate_masks(mask_config, model_tup, images_tup, cuda):
    if mask_config is None:
        mask_config = get_default_mask_config()
    mask_model = MASK(cuda)
    img_x, img_y = images_tup[:2]
    batch_size = mask_config['batch_size']
    num_batches = (len(img_x) + batch_size - 1) // batch_size

    masks = []
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min(len(img_x), start_index + batch_size)
        bx, by = img_x[start_index:end_index], img_y[start_index:end_index]
        masks.append(generate_mask_per_batch(mask_config, mask_model, model_tup, (bx, by), cuda).detach().cpu().numpy())

    return np.concatenate(masks, axis=0)


def generate_mask_per_batch_v2(mask_config, mask_model, model_tup, batch_tup, cuda, m_init=None):
    bx, by = batch_tup
    batch_size = len(bx)
    if not isinstance(bx, torch.Tensor):
        bx = torch.tensor(bx)
    if not isinstance(by, torch.Tensor):
        by = torch.tensor(by)
    model, pre_fn, shape = model_tup
    if m_init is None:
        m_init = torch.zeros(batch_size, 1, 28, 28).fill_(0.5)
    else:
        m_init = m_init.clone().detach()
    if cuda:
        bx, by = bx.cuda(), by.cuda()
        m_init = m_init.cuda()
    m_init.requires_grad = True
    optimizer = Adam([m_init], lr=mask_config['lr'])
    bx = F.interpolate(bx, (224 + 4, 224 + 4), mode='bilinear')
    bx_blurred = mask_model.blur1(bx)
    for i in range(mask_config['n_iters']):
        tot_loss = mask_iter_v2(mask_model, model, pre_fn, bx, by,
                                m_init, mask_config['l1_lambda'], mask_config['tv_lambda'],
                                noise_std=mask_config['noise_std'], x_blurred=bx_blurred)[0]
        if mask_config['verbose'] and i % 50 == 0:
            print(i, np.asscalar(tot_loss) / batch_size)
        optimizer.zero_grad()
        tot_loss.backward()
        optimizer.step()
        m_init.data.clamp_(0, 1)
    return m_init


def generate_masks_v2(mask_config, model_tup, images_tup, cuda):
    if mask_config is None:
        mask_config = get_default_mask_config()
    mask_model = MASKV2(cuda)
    img_x, img_y = images_tup[:2]
    batch_size = mask_config['batch_size']
    num_batches = (len(img_x) + batch_size - 1) // batch_size

    masks = []
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min(len(img_x), start_index + batch_size)
        bx, by = img_x[start_index:end_index], img_y[start_index:end_index]
        masks.append(generate_mask_per_batch_v2(mask_config, mask_model, model_tup, (bx, by), cuda).detach().cpu().numpy())

    return np.concatenate(masks, axis=0)
