#!/usr/bin/env python
import os
import argparse

import numpy as np
import cv2
import visdom


from ia_utils.data_utils import load_imagenet_labels

# python expr_vis/visua;oze_rts_benign.py /home/xinyang/Codes/intattack/qqqqq.npz

def resize(img, new_size=(112, 112)):
    img = np.uint8(255 * img.transpose([1, 2, 0]))
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
    img = img.transpose([2, 0, 1])
    return np.float32(img / 255.)


def main(config):
    vis = visdom.Visdom(env='exprrtsbenignvis',port=7778)

    # data_arx = np.load(config.data_path)
    rts_arx = np.load(config.rts_path)
    img_x, img_y, img_yt = (rts_arx['img_x'].copy(), rts_arx['img_y'].copy(),
                                      rts_arx['img_yt'].copy())
    rts_benign_y = rts_arx['saliency_benign_y']
    rts_benign_yt = rts_arx['saliency_benign_yt']
    imagenet_labels = load_imagenet_labels()

    indices = np.random.RandomState(config.seed).choice(len(img_x), size=20, replace=False)
    for index in indices:
        img, y, yt, path = img_x[index], img_y[index], img_yt[index], "si"
        vis.text('path: %s, true label: %s, target label: %s'
                 % (os.path.basename(path), imagenet_labels[y][1], imagenet_labels[yt][1]),
                 win='info_%d' % index, opts=dict(title='info %d' % index))
        rts_y = rts_benign_y[index]
        rts_yt = rts_benign_yt[index]
        vis.images([resize(img), resize(np.repeat(rts_y, 3, 0)), resize(np.repeat(rts_yt, 3, 0))], win='img_%d' % index,
                   opts=dict(title='img %d' % index))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('data_path')
    parser.add_argument('rts_path')
    parser.add_argument('-s', '--seed', type=int, dest='seed')

    config = parser.parse_args()

    print('Please check the configuration', config)
    main(config)
