#!/usr/bin/env python
import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd as autograd


def imagenet_resize_postfn(grad):
    grad = grad.abs().max(1, keepdim=True)[0]
    grad = F.avg_pool2d(grad, 4).squeeze(1)
    shape = grad.shape
    grad = grad.view(len(grad), -1)
    grad_min = grad.min(1, keepdim=True)[0]
    grad = grad - grad_min
    grad_max = grad.max(1, keepdim=True)[0]
    grad = grad / torch.max(grad_max, torch.tensor([1e-8], device='cuda'))
    return grad.view(*shape)


def generate_gs_per_batches(model_tup, bx, by, post_fn=None, keep_grad=False):
    model, pre_fn = model_tup[:2]
    bxp = pre_fn(bx)
    logit = model(bxp)
    loss = F.nll_loss(F.log_softmax(logit), by)
    grad = autograd.grad([loss], [bx], create_graph=keep_grad)[0]
    if post_fn is not None:
        grad = post_fn(grad)
    return grad


def generate_gs(model_tup, x, y, post_fn=None, keep_grad=False, batch_size=48, device='cuda'):
    n = len(x)
    n_batches = (n + batch_size - 1) // batch_size
    generated = []
    for i in range(n_batches):
        si = i * batch_size
        ei = min(n, si + batch_size)
        bx, by = x[si:ei], y[si:ei]
        bx, by = torch.tensor(bx, device=device, requires_grad=True), torch.tensor(by, device='cuda')
        generated.append(generate_gs_per_batches(
            model_tup, bx, by, post_fn=post_fn,
            keep_grad=keep_grad).detach().cpu().numpy())
    generated = np.concatenate(generated, axis=0)
    return generated
