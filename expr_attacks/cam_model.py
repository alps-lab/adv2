#!/usr/bin/env python
import numpy as np
import torch


class CAM(object):

    def __init__(self):
        pass

    def __call__(self, model_tup, forward_tup, x, y=None):
        return cam_forward(model_tup, forward_tup, x, y)


def cam_forward(model_tup, forward_tup, x, y):
    forward_fn, fc_weight_fn = forward_tup
    batch_size = x.size(0)
    cuda = x.is_cuda
    if y is None:
        with torch.no_grad():
            logits = forward_fn(model_tup, x)[-1]
            logits = logits.cpu().numpy()[0]
        true_label = int(np.argmax(logits))
        y = torch.tensor([true_label])
        if cuda:
            y = y.cuda()

    vs, gs, logits = forward_fn(model_tup, x)
    wc = fc_weight_fn(model_tup)[y].view(batch_size, -1, 1, 1)
    prod = (wc * vs).sum(1, keepdim=True)

    return logits, prod
