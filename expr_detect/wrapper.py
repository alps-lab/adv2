import torch
import torch.nn.functional as F

from functools import lru_cache
from expr_detect.squeeze_utils import *


def predict_batch(model, pre_fn, bx_np, cuda):
    bx = torch.tensor(bx_np)
    if cuda:
        bx = bx.cuda()
    with torch.no_grad():
        probs = F.softmax(model(pre_fn(bx)), 1)
    return probs.detach().cpu().numpy()


class Wrapper(object):

    def __init__(self, model, pre_fn, bx, by, batch_size, cuda):
        self.model = model
        self.pre_fn = pre_fn
        self.bx = bx
        self.by = by
        self.batch_size = batch_size
        self.n = len(bx)
        self.n_batches = (self.n + batch_size - 1) // batch_size
        self.cuda = cuda

    @lru_cache(100)
    def get_clean_probs(self, batch_num):
        si = batch_num * self.batch_size
        ei = min(si + self.batch_size, self.n)
        return predict_batch(self.model, self.pre_fn, self.bx[si:ei], self.cuda)

    @lru_cache(100)
    def get_data_depth_probs(self, batch_num, bit_depth):
        si = batch_num * self.batch_size
        ei = min(si + self.batch_size, self.n)
        bx_depth_np = bit_depth_py(self.bx[si:ei], bit_depth)
        bx_depth_np = np.float32(np.uint8(bx_depth_np * 255) / 255.)
        return predict_batch(self.model, self.pre_fn, bx_depth_np, self.cuda)

    @lru_cache(100)
    def get_median_smoothing_probs(self, batch_num, args):
        si = batch_num * self.batch_size
        ei = min(si + self.batch_size, self.n)
        bx_msmooth_np = median_filter_py(self.bx[si:ei], *args)
        bx_msmooth_np = np.float32(np.uint8(bx_msmooth_np * 255) / 255.)
        return predict_batch(self.model, self.pre_fn, bx_msmooth_np, self.cuda)

    @lru_cache(100)
    def get_nonlocal_mean_probs(self, batch_num, args):
        si = batch_num * self.batch_size
        ei = min(si + self.batch_size, self.n)
        search_window, patch_size, strength = args
        bx_nonlocal_mean_np = non_local_means_color_py(self.bx[si:ei],
                                                       search_window=search_window,
                                                       patch_size=patch_size, strength=strength)
        bx_nonlocal_mean_np = np.float32(np.uint8(bx_nonlocal_mean_np * 255) / 255.)
        by_nonlocal_mean_np = predict_batch(self.model, self.pre_fn, bx_nonlocal_mean_np, self.cuda)
        return by_nonlocal_mean_np


def get_batch_stats(wrapper, batch_number, bit_depth, median_smoothing, nonlocal_mean):
    bprobs_np = wrapper.get_clean_probs(batch_number)
    n = len(bprobs_np)
    max_dists = np.zeros((n,), np.float32)

    if bit_depth is not None:
        by_depth_np = wrapper.get_data_depth_probs(batch_number, bit_depth)
        dists = np.linalg.norm(by_depth_np - bprobs_np, 1, 1)
        max_dists = np.maximum(dists, max_dists)
    if median_smoothing is not None:
        by_msmooth_np = wrapper.get_median_smoothing_probs(batch_number, median_smoothing)
        dists = np.linalg.norm(by_msmooth_np - bprobs_np, 1, 1)
        max_dists = np.maximum(dists, max_dists)
    if nonlocal_mean is not None:
        by_nonlocal_mean_np = wrapper.get_nonlocal_mean_probs(batch_number, nonlocal_mean)
        dists = np.linalg.norm(by_nonlocal_mean_np - bprobs_np, 1, 1)
        max_dists = np.maximum(dists, max_dists)
    return max_dists.astype(np.float32)


def compute_threshold(trained_model, percentile=0.95):
    d = {}
    for key, item in trained_model.items():
        dist_sort = np.sort(item)
        n = len(dist_sort)
        d[key] = dist_sort[int(percentile * n)]

    return d
