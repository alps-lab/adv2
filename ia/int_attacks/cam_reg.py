#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import visdom
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from ia.int_models.cam import CAM
from ia_utils.data_utils import (
    transform_images, TEST_TRANSFORM_NO_NORMALIZE, imagenet_denormalize, imagenet_normalize,
    load_imagenet_labels, mask_to_image, cam_to_image)


class CAMRegAttack(object):

    def __init__(self, cam_model):
        self.cam_model = cam_model

    def attack(self, x, y=None, eps=0.01, adv_coeff=2. / 255, reg_coeff=0.07, max_iters=1000, reg_after=300,
                 verbose=False):
        """

        :param x: torch.tensor (c, h, w) [0, 1] without normalization
        :param y: int or None
        :param eps:
        :param adv_coeff:
        :param reg_coeff:
        :param max_iters:
        :param reg_after:
        :param verbose:
        :return:
        """
        assert eps > 0
        assert max_iters > 0
        assert reg_after > 0

        out = {"success": False}

        x = torch.tensor(x[None]).cuda()

        logits, mask = self.cam_model(imagenet_normalize(x))
        logits = logits.detach().cpu().numpy()[0]
        m = mask_to_image(mask[0], resize=(224, 224))
        out["init_logits"] = logits
        out["init_mask"] = mask[0].detach().cpu().numpy()
        out["init_mask_image"] = m
        out["init_cam_image"] = cam_to_image(x[0].detach().cpu().numpy(), m)
        out["success"] = True

        true_label, target_label = int(np.argmax(logits)), int(np.argmin(logits))
        if y is not None:
            target_label = y
        out["true_label"] = true_label
        out["target_label"] = target_label
        if verbose:
            print("true_label:", true_label, "target_label:", target_label)

        cx = torch.tensor(x, requires_grad=True)
        mask_init = torch.tensor(mask)

        best_mask, best_cx, best_int_loss_value = None, None, np.inf
        for step in range(max_iters):
            y = torch.tensor([target_label]).long().cuda()
            logits, mask = self.cam_model(imagenet_normalize(cx), y)
            current_label = np.asscalar(torch.max(logits, 1)[1])
            adv_loss = F.nll_loss(logits, y)
            int_loss = (mask - mask_init).abs().sum()

            adv_loss_value, int_loss_value = (np.asscalar(adv_loss.detach().cpu().numpy()),
                                              np.asscalar(int_loss.detach().cpu().numpy()))

            adv_grad = autograd.grad([adv_loss], [cx], retain_graph=True)[0]
            adv_grad_norm = torch.norm(adv_grad)

            if current_label == target_label and int_loss_value < best_int_loss_value:
                best_mask = torch.tensor(mask)
                best_cx = torch.tensor(cx)
                best_int_loss_value = int_loss_value

            if step >= reg_after:
                int_grad = autograd.grad([int_loss], [cx], retain_graph=True)[0]
                int_grad_norm = torch.norm(int_grad)
            else:
                int_grad = 0
                int_grad_norm = 1

            with torch.no_grad():
                cx = cx - adv_coeff * adv_grad / adv_grad_norm - reg_coeff * int_grad / int_grad_norm
                diff = cx - x
                diff.clamp_(-eps, eps)
                cx = x + diff
                cx.clamp_(0, 1)
            cx = torch.tensor(cx, requires_grad=True)

            if verbose and step % 50 == 0:
                print("step:", step, "current_label:", current_label,
                      "adv_loss:", adv_loss_value, "adv_grad_norm:",
                      np.asscalar(adv_grad_norm.detach().cpu().numpy()),
                      "init_loss:", np.asscalar(int_loss.detach().cpu().numpy()))
                if step >= 300:
                    print(">>> int_grad_norm", np.asscalar(int_grad_norm.detach().cpu().numpy()))

        if best_int_loss_value < np.inf:
            if verbose:
                print("found")
            out["success"] = True
            out["adv_x"] = best_cx.detach().cpu().numpy()
            best_mask = best_mask[0].detach().cpu().numpy()
            out["adv_mask"] = best_mask
            out["adv_mask_image"] = mask_to_image(best_mask, resize=(224, 224))
            out["adv_cam_image"] = cam_to_image(out["adv_x"][0], out["adv_mask_image"])
        elif verbose:
            print("failed")

        return out
