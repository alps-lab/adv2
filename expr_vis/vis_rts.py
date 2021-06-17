#!/usr/bin/env python

# CUDA_VISIBLE_DEVICES=0 python expr_vis/vis_rts.py /home/xinyang/Data/intattack/rev1/target_maps/rts_resnet50/fold_1.npz /home/ningfei/xinyang/data_fix1/target/rts_resnet50/fold_1.npz -s 100

# CUDA_VISIBLE_DEVICES=0 python expr_vis/vis_rts.py /home/xinyang/Data/intattack/rev1/target_maps/rts_densenet169/fold_1.npz /home/ningfei/xinyang/data_fix1/target/rts_densenet169/fold_1.npz -s 100
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
    vis = visdom.Visdom(env='den_rts_target', port=7777)
    # data_arx = np.load(config.data_benign_path)
    rts_arx = np.load(config.rts_benign_path)
    img_x, img_y, img_yt, img_path = (rts_arx['img_x'].copy(), rts_arx['img_y'].copy(),
                                      rts_arx['img_yt'].copy(), rts_arx['img_path'].copy())
    rts_benign_y = rts_arx['target_rts']

    attacked_arx = np.load(config.attacked_path)
    tar_img = rts_arx['target_images'].copy()
    # tar_map = rts_arx['']
    # rts_benign_path

    # target_images
    vis_steps = [1200]
    arrs_imgs = [attacked_arx['pgd_s2_step_%d_adv_x' % vis_step].copy() for vis_step in vis_steps]
    arrs_logits = [attacked_arx['pgd_s2_step_%d_adv_logits' % vis_step].copy() for vis_step in vis_steps]
    arrs_preds = [np.argmax(arrs_logit, 1) for arrs_logit in arrs_logits]
    arrs_flags = [(arrs_pred == img_yt).astype(np.int64) for arrs_pred in arrs_preds]
    # arrs_flags = [data_adv_arx['pgd_step_%d_succeed' % vis_step].copy() for vis_step in vis_steps]
    arrs_rts = [attacked_arx['pgd_s2_step_%d_adv_rts' % vis_step].copy() for vis_step in vis_steps]
    bad_image = np.zeros((3, 224, 224), dtype=np.float32)
    indices = np.random.RandomState(config.seed).choice(len(img_x), size=100, replace=False)

    for i in range(100):
        m2 = []
        m4 = []
        index = indices[i]
        img, y, yt, path = img_x[index], img_y[index], img_yt[index], img_path[index]
        imgs, texts, rts = [], [], []
        # tar_img = []
        t = []
        tar_i = tar_img[index]
        # tar = tar_img[index]
        # imgs.append(resize(img_x[index]))
        texts.append('benign')
        # rts.append(resize(np.repeat(rts_benign_y[index], 3, 0)))

        for j in range(len(arrs_imgs)):
            img, flag = arrs_imgs[j][index], arrs_flags[j][index]
            if flag == 0:
                # t.append(bad_image)
                imgs.append(bad_image)
                texts.append('')
                rts.append(bad_image)
            else:
                # t.append(resize(tar_img[j][index]))
                imgs.append(resize(arrs_imgs[j][index]))
                dis = np.linalg.norm((img_x[index] - img).flatten(), np.inf)
                texts.append('%d steps, %.4f' % (vis_steps[j], dis))
                rts.append(resize(np.repeat(arrs_rts[j][index], 3, 0)))

        for j, (img, text) in enumerate(zip(imgs, texts)):

            m1 = np.uint8(255 * cv2.resize(rts[j][0], (224, 224), interpolation=cv2.INTER_LINEAR))
            m1 = cv2.applyColorMap(m1, cv2.COLORMAP_JET)
            m1 = np.float32(m1 / 255.).transpose([2, 0, 1])[::-1]
            mp = m1
            m1 = (img + m1)
            # m3 = (tar + m1)

            m1 = m1 / m1.max()
            m2.append(m1)
            # m4.append(m3)
            imgs[j] = img


        rts_y = tar_img[index]
        rts_yt = tar_img[index]
        # print(rts_benign_y[index].shape)
        m3 = np.uint8(255 * cv2.resize(np.repeat(rts_benign_y[index,0],3,0), (224, 224), interpolation=cv2.INTER_LINEAR))
        m3 = cv2.applyColorMap(m3, cv2.COLORMAP_JET)
        m3 = np.float32(m3 / 255.).transpose([2, 0, 1])[::-1]
        m3 = (tar_img[index] + m3)
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

        vis.images(rts_y,
                   win='acid_rts_adv_tar_%d' % index,
                   opts=dict(title='acid_rts_tar_%d, path: %s, true label: %s, target label: %s' % (index, os.path.basename(path), imagenet_labels[y][1], imagenet_labels[yt][1])))
        vis.images(m3,
                   win='acid_rts_adv_tar_map_%d' % index,
                   opts=dict(title='acid_rts_adv_tar_map_%d, path: %s, true label: %s, target label: %s' % (index, os.path.basename(path), imagenet_labels[y][1], imagenet_labels[yt][1])))

        vis.images(imgs, win='acid_rts_adv_%d' % index, opts=dict(title='acid_rts_adv_%d,  true label: %s, target label: %s' % (index, imagenet_labels[y][1], imagenet_labels[yt][1])))
        vis.images(m2, win='acid_rts_map_adv_%d' % index, opts=dict(title='acid_rts_map_adv_%d, true label: %s, target label: %s' % (index, imagenet_labels[y][1], imagenet_labels[yt][1])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('data_benign_path')
    parser.add_argument('rts_benign_path')
    parser.add_argument('attacked_path')
    parser.add_argument('-s', '--seed', type=int, dest='seed')

    config = parser.parse_args()

    print('Please check the configuration', config)
    main(config)
