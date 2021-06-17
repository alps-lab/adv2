#!/usr/bin/env python
import numpy as np
import cv2

import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F

from ia_utils.data_utils import imagenet_normalize
from ia_utils.model_utils import gaussian_blur


def tv_norm(x, beta=2.):
    assert x.size(0) == 1
    img = x[0]
    dy = -img[:, :-1] + img[:, 1:]
    dx = (img[:, :, 1:] - img[:, :, :-1]).transpose(1, 2)
    return (dx.pow(2) + dy.pow(2)).pow(beta / 2.).sum()


class MASK(object):

    def __init__(self, model):
        self.model = model
        self.model.cuda()
        self.model.eval()

    def forward(self, m, x, x_blurred, noise, y, tv_beta):
        """

        :param x: torch.tensor (1, c, h, w) [0, 1] RGB
        :param x_blurred:
        :param m:
        :param noise: torch.tensor (1, c, h, w) [0, 1] RGB
        :param y: int
        :param tv_beta: float
        :return:
        """
        perturbed_input = m * x + (1. - m) * x_blurred
        perturbed_input = perturbed_input + noise

        outputs =F.softmax(self.model(imagenet_normalize(perturbed_input)), 1)
        l1_loss = torch.mean(torch.abs(1 - m))
        tv_loss = tv_norm(m, tv_beta)
        class_loss = outputs[0, y]

        return l1_loss, tv_loss, class_loss


    @staticmethod
    def blur_image(x, blur_size=11, blur_sigma=10):
        """

        :param x: np.ndarray (c, h, w) [0, 1] RGB
        :param blur_size: number
        :param blur_sigma: number
        :return: np.ndarray (c, h, w) [0, 1] RGB
        """
        x = x.transpose([1, 2, 0])
        blurred = cv2.GaussianBlur(x, (blur_size, blur_size), blur_sigma)
        return blurred.transpose([2, 0, 1])

    def __call__(self, x, y=None, input_size=227,
                 learning_rate=1e-1, epochs=500,
                 neuron_selection="max", l1_lambda=1e-2, tv_lambda=1e-4,
                 tv_beta=3, blur_size=11, blur_sigma=10, mask_size=28, noise_std=0,
                 verbose=False):
        """

        :param x: np.ndarray (c, h, w) [0, 1] RGB
        :param x_blurred: None or np.ndarray (c, h, w) [0, 1] RGB
        :param y: int or None
        :param input_size:
        :param learning_rate:
        :param epochs:
        :param neuron_selection:
        :param l1_lambda:
        :param tv_lambda:
        :param tv_beta:
        :param blur_size:
        :param blur_sigma:
        :param mask_size:
        :param noise_std:
        :return: (1, c, h, w) ?
        """
        x = cv2.resize(x.transpose([1, 2, 0]), (input_size, input_size))
        x_blurred = MASK.blur_image(x.transpose([2, 0, 1]), blur_size, blur_sigma)

        x = x.transpose([2, 0, 1])
        # x_blurred = x_blurred.transpose([2, 0, 1])

        mask_init = 0.5 * np.ones((mask_size, mask_size), np.float32)
        mask = torch.tensor(torch.tensor(mask_init[None, None]).cuda(), requires_grad=True)

        upsampler = nn.Upsample(size=(input_size, input_size), mode="bilinear").cuda()
        optimizer = Adam([mask], lr=learning_rate, amsgrad=True)

        x = torch.tensor(x[None]).cuda()
        x.requires_grad = True
        x_blurred = gaussian_blur(x, 11, 10).detach()
        # x_blurred = torch.tensor(x_blurred[None]).cuda()

        with torch.no_grad():
            logits = self.model(imagenet_normalize(x))
        target_label = int(np.argmax(logits[0]))
        if y is not None:
            target_label = int(y)

        if verbose:
            print("start training for target label:", target_label)

        losses = []

        for i in range(epochs):
            upsample_mask = upsampler(mask)
            upsample_mask = upsample_mask.expand(1, 3, upsample_mask.size(2), upsample_mask.size(3))

            noise = np.zeros((input_size, input_size, 3), np.float32)
            if noise_std != 0:
                noise = noise + cv2.randn(noise, 0, noise_std)
            noise = torch.tensor(noise.transpose([2, 0, 1])[None]).cuda()

            l1_loss, tv_loss, class_loss = self.forward(upsample_mask, x, x_blurred, noise, target_label, tv_beta)
            tot_loss = l1_lambda * l1_loss + tv_lambda * tv_loss + class_loss

            optimizer.zero_grad()
            tot_loss.backward(retain_graph=True)
            optimizer.step()

            with torch.no_grad():
                mask.clamp_(0, 1)

            l1_loss_value = l1_loss.detach().cpu().numpy()
            tv_loss_value = tv_loss.detach().cpu().numpy()
            class_loss_value = class_loss.detach().cpu().numpy()
            losses.append((l1_loss_value, tv_loss_value, class_loss_value))

            if verbose and i % 25 == 0:
                print('Epoch %d\tL1 Loss %f\tTV Loss %f\tClass Loss %f\tTot Loss %f\t'
                      % (i, l1_loss_value, tv_loss_value,
                         class_loss_value,
                         tot_loss.detach().cpu().numpy()))

        # upsample_mask = upsampler(mask)
        return mask, losses

