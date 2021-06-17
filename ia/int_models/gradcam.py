import numpy as np
import cv2

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from ia_utils.model_utils import GuidedBackpropReLU
from ia_utils.data_utils import imagenet_normalize


class FeatureExtractor(object):

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers

    def __call__(self, x):
        outputs = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                outputs += [x]

        return outputs, x


class GradCam(object):

    def __init__(self, model, target_layer_names):
        self.model = model
        self.model.eval()
        self.model.cuda()

        self.extractor = FeatureExtractor(self.model.features, target_layer_names)

    def forward(self, x):
        feature, x = self.extractor(x)
        x = x.view(x.size(0), -1)
        x = self.model.classifier(x)
        return feature, x

    def __call__(self, x, y=None):
        """

        :param x: tensor.tensor (1, c, h, w) [0, 1]
        :param index: int or None
        :return: tensor.tensor (1, nc) | (1, fc,) | (1, fh, fw)
        """
        features, logits = self.forward(imagenet_normalize(x))

        if y is None:
            y = np.argmax(logits.detach().cpu().numpy())

        # shape: (,)
        target_logit = logits[0, y]
        target = features[-1]
        grads_val = autograd.grad([target_logit],[target], retain_graph=True)[0]

        # shape: (fc, )
        weights = grads_val.mean(2).mean(2)[0]

        # shape: (fc, fh, fw)
        target = target[0]

        # shape: (fh, fw)
        cam = (weights[:, None, None] * target).sum(0)

        cam = F.relu(cam)

        # shape: (1, nc), (1, fc,), (1, fc, fh, fw), (1, fh, fw)
        return logits, weights[None], target[None], cam.expand(1, -1, -1)


class GuidedBackpropReLUModel(object):

    def __init__(self, model):
        self.model = model

        for name, module in model.features._modules.items():
            if isinstance(module, nn.ReLU):
                model.features._modules[name] = GuidedBackpropReLU.apply

    def forward(self, x):
        return self.model(x)

    def __call__(self, x, index=None):
        output = self.forward(x.cuda())

        if index is None:
            index = np.argmax(output.detach().cpu().numpy())

        one_hot = np.zeros((1, output.size(-1)), dtype=np.foat32)
        one_hot[0, index] = 1
        one_hot = torch.tensor(one_hot).cuda()
        one_hot.requires_grad = True

        one_hot = torch.sum(one_hot * output)
        one_hot.backward()

        output = x.grad.detach().cpu().numpy()
        output = output[0]

        return output
