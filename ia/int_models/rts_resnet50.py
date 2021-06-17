#!/usr/bin/env python
"""
 One should care we normalize image to [-1, 1] for this model!
"""
from __future__ import print_function, division

import os

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.model_zoo as model_zoo
from torch.nn.init import normal_


class ResNetEncoder(torchvision.models.ResNet):

    def forward(self, x):
        s0 = x
        x = self.conv1(s0)
        x = self.bn1(x)
        s1 = self.relu(x)
        x = self.maxpool(s1)

        s2 = self.layer1(x)
        s3 = self.layer2(s2)
        s4 = self.layer3(s3)

        s5 = self.layer4(s4)

        x = self.avgpool(s5)
        sX = x.view(x.size(0), -1)
        sC = self.fc(sX)

        return s0, s1, s2, s3, s4, s5, sX, sC


def resnet50encoder(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetEncoder(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    return model


class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, bottleneck_ratio=4,
                 activation_fn=lambda: torch.nn.ReLU(inplace=False)):
        super(Bottleneck, self).__init__()
        bottleneck_channels = out_channels // bottleneck_ratio
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.activation_fn = activation_fn()

        if stride != 1 or in_channels != out_channels:
            self.residual_transformer = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True)
        else:
            self.residual_transformer = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation_fn(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation_fn(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.residual_transformer is not None:
            residual = self.residual_transformer(residual)
        out += residual

        out = self.activation_fn(out)
        return out


def simple_cnn_block(in_channels, out_channels,
                     kernel_size=3, layers=1, stride=1,
                     follow_with_bn=True, activation_fn=lambda: nn.ReLU(True), affine=True):
    assert layers > 0 and kernel_size % 2 > 0 and stride > 0
    current_channels = in_channels
    _modules = []
    for layer in range(layers):
        _modules.append(nn.Conv2d(current_channels, out_channels, kernel_size=kernel_size,
                                  stride=stride if layer == 0 else 1,
                                  padding=kernel_size // 2, bias=not follow_with_bn))
        current_channels = out_channels
        if follow_with_bn:
            _modules.append(nn.BatchNorm2d(current_channels, affine=affine))
        if activation_fn is not None:
            _modules.append(activation_fn())
    return nn.Sequential(*_modules)


def bottleneck_block(in_channels, out_channels, stride=1, layers=1,
                     activation_fn=lambda: torch.nn.ReLU(inplace=False)):
    assert layers > 0 and stride > 0
    current_channels = in_channels
    _modules = []
    for layer in range(layers):
        _modules.append(Bottleneck(current_channels, out_channels, stride=stride if layer == 0 else 1,
                                   activation_fn=activation_fn))
        current_channels = out_channels
    return nn.Sequential(*_modules) if len(_modules) > 1 else _modules[0]


class PixelShuffleBlock(nn.Module):

    def forward(self, x):
        return F.pixel_shuffle(x, 2)


def simple_upsampler_subpixel(in_channels, out_channels, kernel_size=3,
                              activation_fn=lambda: torch.nn.ReLU(inplace=True),
                              follow_with_bn=True):
    _modules = [
        simple_cnn_block(in_channels, out_channels * 4, kernel_size=kernel_size,
                         follow_with_bn=follow_with_bn),
        PixelShuffleBlock(),
        activation_fn(),
    ]
    return nn.Sequential(*_modules)


class UNetUpsampler(nn.Module):

    def __init__(self, in_channels, out_channels, passthrough_channels, follow_up_residual_blocks=1,
                 upsampler_block=simple_upsampler_subpixel, upsampler_kernel_size=3, activation_fn=lambda: torch.nn.ReLU(inplace=False)):
        super(UNetUpsampler, self).__init__()
        assert follow_up_residual_blocks >= 1
        assert passthrough_channels >= 1
        self.upsampler = upsampler_block(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=upsampler_kernel_size,
                                         activation_fn=activation_fn)
        self.follow_up = bottleneck_block(out_channels + passthrough_channels, out_channels,
                                    layers=follow_up_residual_blocks, activation_fn=activation_fn)

    def forward(self, inp, passthrough):
        upsampled = self.upsampler(inp)
        upsampled = torch.cat([upsampled, passthrough], 1)
        return self.follow_up(upsampled)


class RTSaliencyModel(nn.Module):

    def __init__(self, encoder, encoder_scales, encoder_base, upsampler_scales,
                 upsampler_base, fix_encoder=True, use_simple_activation=False,
                 allow_selector=False, num_classes=1000):
        super(RTSaliencyModel, self).__init__()

        self.encoder = encoder
        self.upsampler_scales = upsampler_scales
        self.encoder_scales = encoder_scales
        self.fix_encoder = fix_encoder
        self.use_simple_activation = use_simple_activation

        down = self.encoder_scales
        modulator_size = []
        for up in reversed(range(self.upsampler_scales)):
            upsampler_chans = upsampler_base * 2 ** (up + 1)
            encoder_chans = encoder_base * 2 ** down
            inc = upsampler_chans if down != encoder_scales else encoder_chans
            modulator_size.append(inc)
            self.add_module("up%d" % up,
                            UNetUpsampler(
                                in_channels=inc,
                                passthrough_channels=encoder_chans // 2,
                                out_channels=upsampler_chans // 2,
                                follow_up_residual_blocks=1,
                                activation_fn=lambda: nn.ReLU(),
                            ))

            down -= 1

        self.to_saliency_chans = nn.Conv2d(upsampler_base, 2, 1)

        self.allow_selector = allow_selector

        if self.allow_selector:
            s = encoder_base * 2 ** encoder_scales
            self.selector_module = nn.Embedding(num_classes, s)
            normal_(self.selector_module.weight, 0, 1. / s ** 0.5)

    def get_trainable_parameters(self):
        all_params = self.parameters()
        if not self.fix_encoder: return set(all_params)
        unwanted = self.encoder.parameters()
        return set(all_params) - set(unwanted) - (set(self.selector_module.parameters() if self.allow_selector
                                                      else set()))

    def forward(self, _images, _selectors=None, pt_store=None, model_confidence=0.):
        out = self.encoder(_images)
        if self.fix_encoder:
            out = [e.detach() for e in out]

        down = self.encoder_scales
        main_flow = out[down]

        if self.allow_selector:
            assert _selectors is not None
            em = torch.squeeze(self.selector_module(_selectors.view(-1, 1)), 1)
            act = torch.sum(main_flow * em.view(-1, 2048, 1, 1), 1, keepdim=True)
            th = torch.sigmoid(act - model_confidence)
            main_flow = main_flow * th

            ex = torch.mean(torch.mean(act, 3), 2)
            exists_logits = torch.cat((-ex / 2., ex / 2.), 1)
        else:
            exists_logits = None

        for up in reversed(range(self.upsampler_scales)):
            assert down > 0
            main_flow = self._modules['up%d' % up](main_flow, out[down - 1])
            down -= 1
        saliency_chans = self.to_saliency_chans(main_flow)

        if self.use_simple_activation:
            return torch.unsqueeze(torch.sigmoid(saliency_chans[:, 0, :, :] / 2), dim=1), exists_logits, out[-1]

        a = torch.abs(saliency_chans[:, 0, :, :])
        b = torch.abs(saliency_chans[:, 1, :, :])
        return torch.unsqueeze(a / (a + b), dim=1), exists_logits, out[-1]

    def minimialistic_restore(self, save_dir):
        # assert self.fix_encoder, 'You should not use this function if you are not using a pre-trained encoder like resnet'

        p = os.path.join(save_dir, 'model-%d.ckpt' % 1)
        if not os.path.exists(p):
            raise FileNotFoundError('Could not find any checkpoint at %s, skipping restore' % p)
        for name, data in torch.load(p, map_location=lambda storage, loc: storage).items():
            self._modules[name].load_state_dict(data)

    def minimalistic_save(self, save_dir):
        assert self.fix_encoder, 'You should not use this function if you are not using a pre-trained encoder like resnet'
        data = {}
        for name, module in self._modules.items():
            if module is self.encoder:  # we do not want to restore the encoder as it should have its own restore function
                continue
            data[name] = module.state_dict()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save(data, os.path.join(save_dir, 'model-%d.ckpt' % 1))


def _gaussian_kernels(kernel_size, sigma, chans):
    assert kernel_size % 2, 'Kernel size of the gaussian blur must be odd!'
    x = np.expand_dims(np.array(range(-kernel_size // 2, -kernel_size // 2 + kernel_size, 1)), 0)
    vals = np.exp(-np.square(x) / (2.*sigma**2))
    _kernel = np.reshape(vals / np.sum(vals), (1, 1, kernel_size, 1))
    kernel = np.zeros((chans, 1, kernel_size, 1), dtype=np.float32) + _kernel
    return kernel, np.transpose(kernel, [0, 1, 3, 2])


def gaussian_blur(_images, kernel_size=55, sigma=11):
    ''' Very fast, linear time gaussian blur, using separable convolution. Operates on batch of images [N, C, H, W].
    Returns blurred images of the same size. Kernel size must be odd.
    Increasing kernel size over 4*simga yields little improvement in quality. So kernel size = 4*sigma is a good choice.'''
    kernel_a, kernel_b = _gaussian_kernels(kernel_size=kernel_size, sigma=sigma, chans=_images.size(1))
    kernel_a = torch.Tensor(kernel_a)
    kernel_b = torch.Tensor(kernel_b)
    if _images.is_cuda:
        kernel_a = kernel_a.cuda()
        kernel_b = kernel_b.cuda()
    _rows = F.conv2d(_images, kernel_a, groups=_images.size(1), padding=(kernel_size // 2, 0))
    return F.conv2d(_rows, kernel_b, groups=_images.size(1), padding=(0, kernel_size // 2))


def apply_mask(images, mask, noise=True, random_colors=True, blurred_version_prob=0.5, noise_std=0.11,
               color_range=0.66, blur_kernel_size=55, blur_sigma=11,
               bypass=0., boolean=False, preserved_imgs_noise_std=0.03):
    images = images.clone()
    cuda = images.is_cuda

    if boolean:
        # remember its just for validation!
        return (mask > 0.5).float() *images

    assert 0. <= bypass < 0.9
    n, c, _, _ = images.size()
    if preserved_imgs_noise_std > 0:
        images = images + torch.empty_like(images).normal_(std=preserved_imgs_noise_std)
    if bypass > 0:
        mask = (1.-bypass)*mask + bypass
    if noise and noise_std:
        alt = torch.empty_like(images).normal_(std=noise_std)
    else:
        alt = torch.zeros_like(images)
    if random_colors:
        if cuda:
            alt += torch.Tensor(n, c, 1, 1).cuda().uniform_(-color_range/2., color_range/2.)
        else:
            alt += torch.Tensor(n, c, 1, 1).uniform_(-color_range/2., color_range/2.)

    if blurred_version_prob > 0.: # <- it can be a scalar between 0 and 1
        cand = gaussian_blur(images, kernel_size=blur_kernel_size, sigma=blur_sigma)
        if cuda:
            when =(torch.Tensor(n, 1, 1, 1).cuda().uniform_(0., 1.) < blurred_version_prob).float()
        else:
            when =(torch.Tensor(n, 1, 1, 1).uniform_(0., 1.) < blurred_version_prob).float()
        alt = alt * (1. - when) + cand * when

    return (mask * images.detach()) + (1. - mask) * alt.detach()


def calc_smoothness_loss(mask, power=2, border_penalty=0.3):
    ''' For a given image this loss should be more or less invariant to image resize when using power=2...
        let L be the length of a side
        EdgesLength ~ L
        EdgesSharpness ~ 1/L, easy to see if you imagine just a single vertical edge in the whole image'''
    x_loss = torch.sum((torch.abs(mask[:, :, 1:, :] - mask[:, :, :-1, :])) ** power)
    y_loss = torch.sum((torch.abs(mask[:, :, :, 1:] - mask[:, :, :, :-1])) ** power)
    if border_penalty > 0:
        border = (float(border_penalty) * torch.sum(mask[:, :, -1, :] ** power +
                                                    mask[:, :, 0, :] ** power +
                                                    mask[:, :, :, -1] ** power + mask[:, :, :, 0]**power))
    else:
        border = 0.
    return (x_loss + y_loss + border) / float(power * mask.size(0))  # watch out, normalised by the batch size!


def calc_area_loss(mask, power=1.):
    if power != 1:
        mask = (mask + 0.0005) ** power # prevent nan (derivative of sqrt at 0 is inf)
    return torch.mean(mask)


def cw_loss(logits, one_hot_labels, targeted=True, t_conf=2, nt_conf=5):
    ''' computes the advantage of the selected label over other highest prob guess.
        In case of the targeted it tries to maximise this advantage to reach desired confidence.
        For example confidence of 3 would mean that the desired label is e^3 (about 20) times more probable than the second top guess.
        In case of non targeted optimisation the case is opposite and we try to minimise this advantage - the probability of the label is
        20 times smaller than the probability of the top guess.
        So for targeted optim a small confidence should be enough (about 2) and for non targeted about 5-6 would work better (assuming 1000 classes so log(no_idea)=6.9)
    '''
    this = torch.sum(logits*one_hot_labels, 1)
    other_best, _ = torch.max(logits*(1.-one_hot_labels) - 12111*one_hot_labels, 1)   # subtracting 12111 from selected labels to make sure that they dont end up a maximum
    t = F.relu(other_best - this + t_conf)
    nt = F.relu(this - other_best + nt_conf)
    if isinstance(targeted, (bool, int)):
        return torch.mean(t) if targeted else torch.mean(nt)
    else:  # must be a byte tensor of zeros and ones

        return torch.mean(t*(targeted>0).float() + nt*(targeted==0).float())


def one_hot(labels, depth):
    if labels.is_cuda:
        return torch.zeros(labels.size(0), depth).cuda().scatter_(1, labels.long().view(-1, 1).data, 1)
    else:
        return torch.zeros(labels.size(0), depth).scatter_(1, labels.long().view(-1, 1).data, 1)


class SaliencyLoss:
    def __init__(self, black_box_fn, area_loss_coef=8, smoothness_loss_coef=0.5, preserver_loss_coef=0.3,
                 num_classes=1000, area_loss_power=0.3, preserver_confidence=1, destroyer_confidence=5, **apply_mask_kwargs):
        self.black_box_fn = black_box_fn
        self.area_loss_coef = area_loss_coef
        self.smoothness_loss_coef = smoothness_loss_coef
        self.preserver_loss_coef = preserver_loss_coef
        self.num_classes = num_classes
        self.area_loss_power =area_loss_power
        self.preserver_confidence = preserver_confidence
        self.destroyer_confidence = destroyer_confidence
        self.apply_mask_kwargs = apply_mask_kwargs

    def get_loss(self, _images, _targets, _masks, _is_real_target=None, pt_store=None):
        ''' masks must be already in the range 0,1 and of shape:  (B, 1, ?, ?)'''
        if _masks.size()[-2:] != _images.size()[-2:]:
            _masks = F.upsample(_masks, (_images.size(2), _images.size(3)), mode='bilinear')

        if _is_real_target is None:
            _is_real_target = torch.ones_like(_targets)
        destroyed_images = apply_mask(_images, 1.-_masks, **self.apply_mask_kwargs)
        destroyed_logits = self.black_box_fn(destroyed_images)

        preserved_images = apply_mask(_images, _masks, **self.apply_mask_kwargs)
        preserved_logits = self.black_box_fn(preserved_images)

        _one_hot_targets = one_hot(_targets, self.num_classes)
        preserver_loss = cw_loss(preserved_logits, _one_hot_targets, targeted=_is_real_target == 1, t_conf=self.preserver_confidence, nt_conf=1.)
        destroyer_loss = cw_loss(destroyed_logits, _one_hot_targets, targeted=_is_real_target == 0, t_conf=1., nt_conf=self.destroyer_confidence)
        area_loss = calc_area_loss(_masks, self.area_loss_power)
        smoothness_loss = calc_smoothness_loss(_masks)

        total_loss = destroyer_loss + self.area_loss_coef*area_loss + self.smoothness_loss_coef*smoothness_loss + self.preserver_loss_coef*preserver_loss

        if pt_store is not None:
            # add variables to the pt_store
            pt_store(masks=_masks)
            pt_store(destroyed=destroyed_images)
            pt_store(preserved=preserved_images)
            pt_store(area_loss=area_loss)
            pt_store(smoothness_loss=smoothness_loss)
            pt_store(destroyer_loss=destroyer_loss)
            pt_store(preserver_loss=preserver_loss)
            pt_store(preserved_logits=preserved_logits)
            pt_store(destroyed_logits=destroyed_logits)
        return total_loss


def to_batch_variable(x, required_rank, cuda=False):
    if isinstance(x, torch.Tensor):
        if cuda and not x.is_cuda:
            return x.cuda()
        if not cuda and x.is_cuda:
            return x.cpu()
        else:
            return x
    if isinstance(x, (float, int)):
        assert required_rank == 1
        return to_batch_variable(np.array([x]), required_rank, cuda)
    if isinstance(x, (list, tuple)):
        return to_batch_variable(np.array(x), required_rank, cuda)
    if isinstance(x, np.ndarray):
        c = len(x.shape)
        if c == required_rank:
            return to_batch_variable(torch.from_numpy(x), required_rank, cuda)
        elif c + 1 == required_rank:
            return to_batch_variable(torch.unsqueeze(torch.from_numpy(x), dim=0), required_rank, cuda)
        else:
            raise ValueError()


def get_pretrained_saliency_fn(ckpt_dir, cuda=True, return_classification_logits=False):
    ''' returns a saliency function that takes images and class selectors as inputs. If cuda=True then places the model on a GPU.
    You can also specify model_confidence - smaller values (~0) will show any object in the image that even slightly resembles the specified class
    while higher values (~5) will show only the most salient parts.
    Params of the saliency function:
    images - input images of shape (C, H, W) or (N, C, H, W) if in batch. Can be either a numpy array, a Tensor or a Variable
    selectors - class ids to be masked. Can be either an int or an array with N integers. Again can be either a numpy array, a Tensor or a Variable
    model_confidence - a float, 6 by default, you may want to decrease this value to obtain more complete saliency maps.
    returns a Variable of shape (N, 1, H, W) with one saliency maps for each input image.
    '''
    saliency = RTSaliencyModel(resnet50encoder(pretrained=True), 5, 64, 3, 64, fix_encoder=False, use_simple_activation=False, allow_selector=True)
    saliency.minimialistic_restore(ckpt_dir)
    saliency.train(False)
    if cuda:
        saliency = saliency.cuda()

    def saliency_fn(images, selectors, model_confidence=6):
        _images, _selectors = to_batch_variable(images, 4, cuda), to_batch_variable(selectors, 1, cuda).long()
        masks, _, cls_logits = saliency(_images * 2, _selectors, model_confidence=model_confidence)
        sal_map = F.upsample(masks, (_images.size(2), _images.size(3)), mode='bilinear')
        if not return_classification_logits:
            return sal_map
        return sal_map, cls_logits

    def logits_fn(images):
        _images = to_batch_variable(images, 4, cuda)
        logits = saliency.encoder(_images * 2)[-1]
        return logits

    return saliency_fn, logits_fn


def read_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = np.transpose(img[..., ::-1], [2, 0, 1])
    img = np.float32(img) / 255. * 2 - 1
    return to_batch_variable(img, 4)
