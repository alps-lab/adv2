#!/usr/bin/env python
import os
import re
import argparse

import numpy as np
import cv2
import visdom


def draw_text(img, text):
    img = cv2.putText(np.uint8(255 * img[::-1].transpose([1, 2, 0])).copy(), text,
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                0)
    return np.float32(img.transpose([2, 0, 1])[::-1] / 255.)


def resize(img, new_size=(112, 112)):
    img = np.uint8(255 * img.transpose([1, 2, 0]))
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
    img = img.transpose([2, 0, 1])
    return np.float32(img / 255.)


def main(config):
    vis = visdom.Visdom(env='exprrtsadvvis')
    data_arx = np.load(config.data_benign_path)
    rts_arx = np.load(config.rts_benign_path)
    img_x, img_y, img_yt, img_path = (data_arx['img_x'].copy(), data_arx['img_y'].copy(),
                                      data_arx['img_yt'].copy(), data_arx['img_path'].copy())
    rts_benign_y = rts_arx['saliency_benign_y']

    data_adv_arx = np.load(config.data_adv_path)
    rts_adv_arx = np.load(config.rts_adv_path)

    vis_steps = [50, 100, 150, 200, 250]
    arrs_imgs = [data_adv_arx['pgd_step_%d_adv_x' % vis_step].copy() for vis_step in vis_steps]
    arrs_flags = [data_adv_arx['pgd_step_%d_succeed' % vis_step].copy() for vis_step in vis_steps]
    arrs_rts = [rts_adv_arx['pgd_step_%d_adv_rts_yt' % vis_step].copy() for vis_step in vis_steps]
    bad_image = np.zeros((3, 112, 112), dtype=np.float32)
    indices = np.random.RandomState(config.seed).choice(len(img_x), size=30, replace=False)

    for i in range(30):
        index = indices[i]
        imgs, texts, rts = [], [], []
        imgs.append(resize(img_x[index]))
        texts.append('benign')
        rts.append(resize(np.repeat(rts_benign_y[index], 3, 0)))

        for j in range(len(arrs_imgs)):
            img, flag = arrs_imgs[j][index], arrs_flags[j][index]
            if flag == 0:
                imgs.append(bad_image)
                texts.append('')
                rts.append(bad_image)
            else:
                imgs.append(resize(arrs_imgs[j][index]))
                dis = np.linalg.norm((img_x[index] - img).flatten(), np.inf)
                texts.append('%d steps, %.4f' % (vis_steps[j], dis))
                rts.append(resize(np.repeat(arrs_rts[j][index], 3, 0)))

        for j, (img, text) in enumerate(zip(imgs, texts)):
            imgs[j] = draw_text(img, text)
        vis.images(np.concatenate([np.stack(imgs), np.stack(rts)], axis=0), nrow=len(imgs) // 2,
                   win='img_%d' % index, opts=dict(title='img %d' % index))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_benign_path')
    parser.add_argument('data_adv_path')
    parser.add_argument('rts_benign_path')
    parser.add_argument('rts_adv_path')
    parser.add_argument('-s', '--seed', type=int, dest='seed')

    config = parser.parse_args()

    print('Please check the configuration', config)
    main(config)
