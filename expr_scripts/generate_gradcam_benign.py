#!/usr/bin/env python
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
    dobj = np.load(config.data_path)
    img_x, img_y, img_yt = dobj['img_x'].copy(), dobj['img_y'].copy(), dobj['img_yt'].copy()
    gradcam_config = dict(batch_size=config.batch_size)
    gradcam_model = load_model(config)
    cuda = config.device == 'gpu'
    gradcams_y, gradcams_y_n = generate_gradcams(gradcam_config, gradcam_model, (img_x, img_y), cuda)
    gradcams_yt, gradcams_yt_n = generate_gradcams(gradcam_config, gradcam_model, (img_x, img_yt), cuda)
    np.savez(config.save_path, gradcam_benign_y=gradcams_y, gradcam_benign_y_n=gradcams_y_n,
             gradcam_benign_yt=gradcams_yt, gradcam_benign_yt_n=gradcams_yt_n)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
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
