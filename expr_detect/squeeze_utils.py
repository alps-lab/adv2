#!/usr/bin/env python
import numpy as np
import torch
from scipy import ndimage
import cv2


def reduce_precision_py(x, npp):
    """
    Reduce the precision of image, the numpy version.
    :param x: a float tensor, which has been scaled to [0, 1].
    :param npp: number of possible values per pixel. E.g. it's 256 for 8-bit gray-scale image, and 2 for binarized image.
    :return: a tensor representing image(s) with lower precision.
    """
    # Note: 0 is a possible value too.
    npp_int = npp - 1
    x_int = np.rint(x * npp_int)
    x_float = x_int / npp_int
    return x_float


def bit_depth_py(x, bits):
    precisions = 2 ** bits
    return reduce_precision_py(x, precisions)


def reduce_precision_torch(x, npp):
    """
    Reduce the precision of image, the numpy version.
    :param x: a float tensor, which has been scaled to [0, 1].
    :param npp: number of possible values per pixel. E.g. it's 256 for 8-bit gray-scale image, and 2 for binarized image.
    :return: a tensor representing image(s) with lower precision.
    """
    # Note: 0 is a possible value too.
    x= x.detach()
    npp_int = npp - 1
    x_int = torch.round(x * npp_int)
    x_float = x_int / npp_int
    return x_float


def bit_depth_torch(x, bits):
    precisions = 2 ** bits
    return reduce_precision_torch(x, precisions)


def median_filter_py(x, height, width=None):
    """
    Median smoothing by Scipy.
    :param x: a tensor of image(s)
    :param width: the width of the sliding window (number of pixels)
    :param height: the height of the window. The same as width by default.
    :return: a modified tensor with the same shape as x.
    """
    if width is None:
        width = height
    return ndimage.filters.median_filter(x, size=(1, 1, height, width), mode='reflect')


def non_local_means_color_py(imgs, search_window, patch_size, strength):
    img_fs = []
    for img in imgs:
        img_u8 = np.uint8(img * 255)
        img_f = cv2.fastNlMeansDenoisingColored(img_u8.transpose([1, 2, 0]), None, h=strength, hColor=strength,
                                                templateWindowSize=patch_size,
                                                searchWindowSize=search_window)
        img_fs.append(np.float32(img_f.transpose([2, 0, 1]) / 255.))
    return np.stack(img_fs, 0)
