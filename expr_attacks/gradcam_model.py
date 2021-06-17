import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from ia_utils.model_utils import GuidedBackpropReLU


class GradCam(object):

    def __init__(self, model_tup, extractor_fn):
        model, pre_fn = model_tup[:2]
        self.model_tup = (model, pre_fn)
        self.extractor = extractor_fn

    def forward(self, x):
        feature, logits = self.extractor(self.model_tup, x)
        return feature, logits

    def __call__(self, x, y, create_graph=False):
        """

        :param x: tensor.tensor (n, c, h, w) [0, 1]
        :param y: tensor.tensor (n, )
        :param index: int or None
        :return: tensor.tensor (n, nc) | (n, fc,) | (n, fh, fw)
        """
        batch_size = x.size(0)
        features, logits = self.forward(x)
        # shape: (n,)
        target_logit = logits.gather(1, y.view(batch_size, -1))[:, 0]
        # shape: (n, fc, fh, fw)
        target = features[-1]
        grads_val = autograd.grad([target_logit.sum()],[target], create_graph=create_graph)[0]

        # shape: (n, fc)
        weights = grads_val.mean(2).mean(2)

        # shape: (n, 1, fh, fw)
        gradcam = (weights[..., None, None] * target).sum(1, keepdim=True)
        gradcam = F.relu(gradcam)

        gradcam_expand = gradcam.view(batch_size, -1)
        gradcam_expand_min = gradcam_expand.min(1, keepdim=True)[0]
        gradcam_expand = gradcam_expand - gradcam_expand_min
        gradcam_expand_max = gradcam_expand.max(1, keepdim=True)[0]
        gradcam_expand = gradcam_expand / gradcam_expand_max
        gradcam_n = gradcam_expand.view(*gradcam.size())

        # shape: (n, nc), (n, fc,), (n, fc, fh, fw), (n, 1, fh, fw), (n, 1, fh, fw)
        return logits, weights, target, gradcam, gradcam_n


def make_guided_backprop_relu_model(model):
    for name, module in model._modules.items():
        if isinstance(module, nn.ReLU):
            model._modules[name] = GuidedBackpropReLU.apply

