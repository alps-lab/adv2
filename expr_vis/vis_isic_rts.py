#!/usr/bin/env python

# CUDA_VISIBLE_DEVICES=0 python expr_vis/vis_rts.py /home/xinyang/Data/intattack/rev1/target_maps/rts_resnet50/fold_1.npz /home/ningfei/xinyang/data_fix1/target/rts_resnet50/fold_1.npz -s 100

# python rev1/isic_acid/rts_attack_acid.py /home/xinyang/Data/intattack/rev1/benign_maps/isic_rts_benign/fold_1.npz /home/xinyang/Data/intattack/rev1/benign_maps/isic_rts_acid/fold_1.npz -c 18

# CUDA_VISIBLE_DEVICES=0 python expr_vis/vis_isic_rts.py /home/xinyang/Data/intattack/rev1/benign_maps/isic_rts_acid/fold_1.npz -s 100
import os
import re
import argparse

import numpy as np
import cv2
import visdom
from ia_utils.data_utils import load_imagenet_labels
imagenet_labels = load_imagenet_labels()


def draw_text(img, text):
    img = cv2.putText(np.uint8(255 * img[::-1].transpose([1, 2, 0])).copy(), text,
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                0)
    return np.float32(img.transpose([2, 0, 1])[::-1] / 255.)


def resize(img, new_size=(224, 224)):
    img = np.uint8(255 * img.transpose([1, 2, 0]))
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
    img = img.transpose([2, 0, 1])
    return np.float32(img / 255.)


def main(config):
    vis = visdom.Visdom(env='isic_rts_reg_reg', port=7777)

    attacked_arx = np.load(config.attacked_path)
    # data_arx = np.load(config.data_benign_path)
    # rts_arx = np.load(config.rts_benign_path)
    img_x, img_y, img_yt = (attacked_arx['img_x'].copy(), attacked_arx['img_y'].copy(),attacked_arx['img_yt'].copy())
    rts_benign_y = attacked_arx['saliency_benign_y']


    tar_img = attacked_arx['saliency_benign_y'].copy()
    # tar_map = rts_arx['']
    # rts_benign_path

    # target_images

    bad_image = np.zeros((3, 224, 224), dtype=np.float32)
    indices = np.random.RandomState(config.seed).choice(len(img_x), size=100, replace=False)

    for i in range(100):
        m2 = []
        m4 = []
        index = indices[i]
        img, y, yt, path = img_x[index], img_y[index], img_yt[index], " "
        imgs, texts, rts = [], [], []
        # tar_img = []
        t = []
        tar_i = tar_img[index]
        # tar = tar_img[index]
        # imgs.append(resize(img_x[index]))
        texts.append('benign')
        # rts.append(resize(np.repeat(rts_benign_y[index], 3, 0)))


        rts_y = tar_img[index]
        rts_yt = tar_img[index]
        # print(rts_benign_y[index].shape)
        m3 = np.uint8(255 * cv2.resize(np.repeat(rts_benign_y[index,0],3,0), (224, 224), interpolation=cv2.INTER_LINEAR))
        m3 = cv2.applyColorMap(m3, cv2.COLORMAP_JET)
        m3 = np.float32(m3 / 255.).transpose([2, 0, 1])[::-1]
        m3 = (img_x[index] + m3)
        m3 = m3 / m3.max()
        # for j, (img, text) in enumerate(zip(t, texts)):
        #
        #     m1 = np.uint8(255 * cv2.resize(rts[j][0], (224, 224), interpolation=cv2.INTER_LINEAR))
        #     m1 = cv2.applyColorMap(m1, cv2.COLORMAP_JET)
        #     m1 = np.float32(m1 / 255.).transpose([2, 0, 1])[::-1]
        #     mp = m1
        #     m1 = (img + m1)
        #     # m3 = (tar + m1)
        #
        #     m1 = m1 / m1.max()
        #     m4.append(m1)
        #     # m4.append(m3)
        #     t[j] = img

        # tar_i_2 = (mp + tar_i)
        # tar_i_2 = tar_i / tar_i_2.max()

        vis.images(img_x[index],
                   win='acid_rts_adv_tar_%d' % index,
                   opts=dict(title='acid_rts_tar_%d, path: %s, true label: %s, target label: %s' % (index, os.path.basename(path), imagenet_labels[y][1], imagenet_labels[yt][1])))
        vis.images(m3,
                   win='acid_rts_adv_tar_map_%d' % index,
                   opts=dict(title='acid_rts_adv_tar_map_%d, path: %s, true label: %s, target label: %s' % (index, os.path.basename(path), imagenet_labels[y][1], imagenet_labels[yt][1])))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('data_benign_path')
    # parser.add_argument('rts_benign_path')
    parser.add_argument('attacked_path')
    parser.add_argument('-s', '--seed', type=int, dest='seed')

    config = parser.parse_args()

    print('Please check the configuration', config)
    main(config)
