import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ia_utils.model_utils import get_gaussian_blur_kernel


def tv_norm(img, beta=2., epsilon=1e-8):
    batch_size = img.size(0)
    dy = -img[:, :, :-1] + img[:, :, 1:]
    dx = (img[:, :, :, 1:] - img[:, :, :, :-1]).transpose(2, 3)
    return (dx.pow(2) + dy.pow(2) + epsilon).pow(beta / 2.).view(batch_size, -1).sum(1)


class GaussianBlur(nn.Module):

    def __init__(self, ksize, sigma, num_channels=3):
        super(GaussianBlur, self).__init__()
        self.ksize = ksize
        self.sigma = sigma
        self.psize = int((ksize - 1) / 2)
        self.num_channels = num_channels
        self.blur_kernel = nn.Parameter(get_gaussian_blur_kernel(ksize, sigma).repeat(num_channels, 1, 1, 1),
                                        requires_grad=False)

    def forward(self, x):
        x_padded = F.pad(x, [self.psize] * 4, mode="reflect")
        return F.conv2d(x_padded, self.blur_kernel, groups=self.num_channels)


class MASK(object):

    def __init__(self, cuda):
        self.gaussian_blur = GaussianBlur(11, 10)
        if cuda:
            self.gaussian_blur.cuda()


def mask_iter(mask_model, model, pre_fn, x, y, r, m_init, l1_lambda=1e-2, tv_lambda=1e-4, tv_beta=3., noise_std=0.,
              weights=None, x_blurred=None):
    batch_size = x.size(0)
    cuda = x.is_cuda
    if r is not None:
        x = x + r
    if x_blurred is None:
        x_blurred = mask_model.gaussian_blur(x)
    m = F.upsample(m_init, size=(x.size(2), x.size(3)), mode="bilinear")
    perturbed_inputs = m * x + (1. - m) * x_blurred
    if noise_std != 0:
        noise = noise_std * torch.randn(*perturbed_inputs.size())
        if cuda:
            noise = noise.cuda()
        perturbed_inputs = perturbed_inputs + noise

    outputs = F.softmax(model(pre_fn(perturbed_inputs)), 1)
    l1_loss = torch.mean(torch.abs(1 - m_init).view(batch_size, -1), 1)
    tv_loss = tv_norm(m_init, tv_beta)
    class_loss = outputs.gather(1, y[:, None])[:, 0]
    if weights is None:
        tot_loss = l1_lambda * torch.sum(l1_loss) + tv_lambda * torch.sum(tv_loss) + torch.sum(class_loss)
    else:
        tot_loss = (l1_lambda * torch.sum(l1_loss * weights) + tv_lambda * torch.sum(tv_loss * weights) +
                    torch.sum(class_loss * weights))
    return tot_loss, [l1_loss, tv_lambda, class_loss]


class MASKV2(object):

    def __init__(self, cuda):
        self.blur1 = GaussianBlur(21, -1)
        self.blur2 = GaussianBlur(11, -1, 1)
        self.cuda = cuda
        if cuda:
            self.blur1.cuda()
            self.blur2.cuda()


def mask_iter_v2(mask_model, model, pre_fn, x, y, m_init, l1_lambda=1e-4, tv_lambda=1e-2, tv_beta=3., noise_std=0.,
                 jitter=4, weights=None, x_blurred=None):
    batch_size = x.size(0)
    if x_blurred is None:
        x_blurred = mask_model.blur1(x)

    if jitter != 0:
        j1 = np.random.randint(jitter)
        j2 = np.random.randint(jitter)
    else:
        j1, j2 = 0, 0
    x_ = x[:, :, j1:j1+224, j2:j2+224]
    x_blurred_ = x_blurred[:, :, j1:j1+224, j2:j2+224]

    if noise_std != 0:
        noisy = torch.randn_like(m_init)
        mask_w_noisy = m_init + noisy
        mask_w_noisy.clamp_(0, 1)
    else:
        mask_w_noisy = m_init

    mask_w_noisy = F.interpolate(mask_w_noisy, (224, 224), mode='bilinear')
    mask_w_noisy = mask_model.blur2(mask_w_noisy)
    x = x_ * mask_w_noisy + x_blurred_ * (1 - mask_w_noisy)

    class_loss = F.softmax(model(pre_fn(x)), dim=-1).gather(1, y.unsqueeze(1)).squeeze(1)
    l1_loss = (1 - m_init).abs().view(batch_size, -1).sum(-1)
    tv_loss = tv_norm(m_init, tv_beta)

    if weights is None:
        tot_loss = l1_lambda * torch.sum(l1_loss) + tv_lambda * torch.sum(tv_loss) + torch.sum(class_loss)
    else:
        tot_loss = (l1_lambda * torch.sum(l1_loss * weights) + tv_lambda * torch.sum(tv_loss * weights) +
                    torch.sum(class_loss * weights))
    return tot_loss, [l1_loss, tv_lambda, class_loss]
