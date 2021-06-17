import torch
import torch.nn as nn


class GradSaliency(object):

    def __init__(self, base_model):
        self.base_model = base_model
        self.base_model.eval()
        self.base_model.cuda()

    def forward(self, x):
        return self.base_model.forward(x)

    def __call__(self, x, y=None):
        bs = x.size(0)
        logits = self.base_model(x)
        if y is None:
            y = torch.max(logits, 1)[1]
        y = logits[torch.arange(0, bs, dtype=torch.long), y]
        w = torch.autograd.grad([y], [x], allow_unused=True, create_graph=True, retain_graph=True)[0]
        m = w.abs().max(1)[0]

        return logits, m
