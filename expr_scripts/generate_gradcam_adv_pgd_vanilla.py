#!/usr/bin/env python
import re
import argparse

import numpy as np
import torch


from expr_attacks.gradcam_model_def import gradcam_resnet50
from expr_attacks.gradcam_generator import generate_gradcams
from expr_attacks.gradcam_model import GradCam
from expr_attacks.gradcam_model import make_guided_backprop_relu_model


def load_model(config):
    model_tup, extractor_fn = gradcam_resnet50()
    model = model_tup[0]
    if config.device == 'gpu':
        model.cuda()
    model.train(False)
    make_guided_backprop_relu_model(model)
    return GradCam(model_tup, extractor_fn)


def main(config):
    pattern = re.compile(r'pgd_step_(?P<step>\d+)_adv_x')
    dobj = np.load(config.data_path)
    dobj_adv = np.load(config.data_path_adv)
    img_yt = dobj['img_yt'].copy()
    keys = list(dobj_adv.keys())
    kept_keys = {}
    for key in keys:
        matchobj = pattern.match(key)
        if matchobj is not None:
            kept_keys[int(matchobj.group('step'))] = key

    gradcam_config = dict(batch_size=config.batch_size)
    gradcam_model = load_model(config)
    robj = {}
    for step, key in kept_keys.items():
        img_x = dobj_adv[key].copy()
        gradcam_adv_yt, gradcam_adv_yt_n = generate_gradcams(gradcam_config, gradcam_model, (img_x, img_yt),
                                                                config.device == 'gpu')
        robj['pgd_step_%d_adv_gradcam_yt' % step] = gradcam_adv_yt
        robj['pgd_step_%d_adv_gradcam_yt_n' % step] = gradcam_adv_yt_n
    np.savez(config.save_path, **robj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('data_path_adv')
    parser.add_argument('save_path')
    parser.add_argument('-b', '--batch-size', type=int, dest='batch_size', default=40)
    parser.add_argument('--device', choices=['gpu', 'cpu'], dest='device')

    config = parser.parse_args()
    if config.device is None:
        if torch.cuda.is_available():
            config.device = 'gpu'
        else:
            config.device = 'cpu'
    print('Please check the configuration', config)
    main(config)
