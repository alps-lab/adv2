import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from expr_attacks.mask_model import tv_norm
from ia_utils.model_utils import get_gaussian_blur_kernel


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


class MASKV2(object):

    def __init__(self, cuda):
        self.blur1 = GaussianBlur(5, -1)
        self.blur2 = GaussianBlur(3, -1, 1)
        self.cuda = cuda
        if cuda:
            self.blur1.cuda()
            self.blur2.cuda()


def get_default_mask_config():
    return dict(lr=0.1, l1_lambda=1e-4, tv_lambda=1e-2, noise_std=0, n_iters=400,
                batch_size=40, verbose=False)


def mask_iter_v2(mask_model, model, pre_fn, x, y, m_init, l1_lambda=1e-4, tv_lambda=1e-2, tv_beta=3., noise_std=0., jitter=2, weights=None, x_blurred=None):
    batch_size = x.size(0)
    if x_blurred is None:
        x_blurred = mask_model.blur1(x)

    if jitter != 0:
        j1 = np.random.randint(jitter)
        j2 = np.random.randint(jitter)
    else:
        j1, j2 = 0, 0
    x_ = x[:, :, j1:j1+32, j2:j2+32]
    x_blurred_ = x_blurred[:, :, j1:j1+32, j2:j2+32]

    if noise_std != 0:
        noisy = torch.randn_like(m_init)
        mask_w_noisy = m_init + noisy
        mask_w_noisy.clamp_(0, 1)
    else:
        mask_w_noisy = m_init

    mask_w_noisy = F.interpolate(mask_w_noisy, (32, 32), mode='bilinear')
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


def generate_mask_per_batch_v2(mask_config, mask_model, model_tup, batch_tup, cuda, m_init=None):
    bx, by = batch_tup
    batch_size = len(bx)
    if not isinstance(bx, torch.Tensor):
        bx = torch.tensor(bx)
    if not isinstance(by, torch.Tensor):
        by = torch.tensor(by)
    model, pre_fn, shape = model_tup
    if m_init is None:
        m_init = torch.zeros(batch_size, 1, 8, 8).fill_(0.5)
    else:
        m_init = m_init.clone().detach()
    if cuda:
        bx, by = bx.cuda(), by.cuda()
        m_init = m_init.cuda()
    m_init.requires_grad = True
    optimizer = Adam([m_init], lr=mask_config['lr'])
    bx = F.interpolate(bx, (32 + 2, 32 + 2), mode='bilinear')
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
