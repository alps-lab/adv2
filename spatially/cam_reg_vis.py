

# CUDA_VISIBLE_DEVICES=0 python expr_vis/vis_cam.py /home/xinyang/Data/intattack/rev1/target_maps/cam_resnet50/fold_1.npz /home/ningfei/xinyang/data_fix1/target/cam_resnet50/fold_1.npz

# CUDA_VISIBLE_DEVICES=0 python expr_vis/vis_cam.py /home/xinyang/Data/intattack/rev1/target_maps/cam_densenet169/fold_1.npz /home/ningfei/xinyang/data_fix1/target/cam_densenet169/fold_1.npz

#CUDA_VISIBLE_DEVICES=0 python spatially/cam_reg_vis.py /home/xinyang/Data/intattack/rev1/acid_stadv_maps/cam_densenet169_benign/fold_2.npz /home/xinyang/Data/intattack/rev1/regular_stadv_maps/cam_densenet169/fold_2.npz

import os
import argparse

import numpy as np
import cv2
import visdom

from ia_utils.data_utils import load_imagenet_labels
# from ia_utils.data_utils import load_imagenet_labels
imagenet_labels = load_imagenet_labels()

vis = visdom.Visdom(env='st_cam_reg_den', port=7778)


def draw_text(img, text):
    img = cv2.putText(np.uint8(255 * img[::-1].transpose([1, 2, 0])).copy(), text,
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                0)
    return np.float32(img.transpose([2, 0, 1])[::-1] / 255.)


def transform_cam(img, cam):
    m1 = np.uint8(255 * cv2.resize(cam[0], (224, 224), interpolation=cv2.INTER_LINEAR))
    m1 = cv2.applyColorMap(m1, cv2.COLORMAP_JET)
    m1 = np.float32(m1 / 255.).transpose([2, 0, 1])[::-1]
    m1 = (img + m1)
    m1 = m1 / m1.max()
    return m1


def resize(img, new_size=(224, 224)):
    img = np.uint8(255 * img.transpose([1, 2, 0]))
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
    img = img.transpose([2, 0, 1])
    return np.float32(img / 255.)


def main(config):
    # dobj = np.load(config.data_path)
    # dobj_adv = np.load(config.data_adv_path)
    dobj_cam_benign = np.load(config.cam_benign_path)
    dobj_cam_adv = np.load(config.cam_adv_path)
    print(list(dobj_cam_adv.keys()))
    img_x, img_y, img_yt = dobj_cam_benign['img_x'].copy(), dobj_cam_benign['img_x'].copy(), dobj_cam_benign['img_x'].copy()

    cam_benign_y = dobj_cam_adv['stadv_cam_yt_n'].copy()
    indices = np.random.RandomState(100).choice(len(img_x), size=100, replace=False)
    imagenet_labels = load_imagenet_labels()
    # tar_img = dobj_cam_adv['img_x'].copy()

    vis_steps = [600]
    # arrs_imgs = [dobj_cam_adv['s2_step_%d_stadv_x' % vis_step].copy() for vis_step in vis_steps]
    # arrs_flags = [dobj_cam_adv['s2_step_%d_stadv_succeed' % vis_step].copy() for vis_step in vis_steps]
    # arrs_cams = [dobj_cam_adv['s2_step_%d_stadv_cam_n' % vis_step].copy() for vis_step in vis_steps]
    bad_image = np.zeros((3, 224, 224), dtype=np.float32)

    for i in range(100):
        m2 = []
        index = indices[i]
        imgs, texts, cams = [], [], []
        img, y, yt = img_x[index], img_y[index], img_yt[index]
        # imgs.append(resize(img_x[index]))
        texts.append('benign')
        # cams.append(resize(np.repeat(cam_benign_y[index], 3, 0)))

        # for j in range(len(arrs_imgs)):
        #     img, flag = arrs_imgs[j][index], arrs_flags[j][index]
        #     if flag == 0:
        #         imgs.append(bad_image)
        #         texts.append('')
        #         cams.append(bad_image)
        #     else:
        #         imgs.append(resize(arrs_imgs[j][index]))
        #         dis = np.linalg.norm((img_x[index] - img).flatten(), np.inf)
        #         texts.append('%d steps, %.4f' % (vis_steps[j], dis))
        #         cams.append(resize(np.repeat(arrs_cams[j][index], 3, 0)))
        #         # cams.append(resize(transform_cam(arrs_imgs[j][index], arrs_cams[j][index])))
        #
        # for j, (img, text) in enumerate(zip(imgs, texts)):
        #     m1 = np.uint8(255 * cv2.resize(cams[j][0], (224, 224), interpolation=cv2.INTER_LINEAR))
        #     m1 = cv2.applyColorMap(m1, cv2.COLORMAP_JET)
        #     m1 = np.float32(m1 / 255.).transpose([2, 0, 1])[::-1]
        #     m1 = (img + m1)
        #     m1 = m1 / m1.max()
        #     m2.append(m1)
        #     imgs[j] = img

        # rts_y = tar_img[index]
        # rts_yt = tar_img[index]
        # print(rts_benign_y[index].shape)
        m3 = np.uint8(255 * cv2.resize(cam_benign_y[index, 0], (224, 224), interpolation=cv2.INTER_LINEAR))
        m3 = cv2.applyColorMap(m3, cv2.COLORMAP_JET)
        m3 = np.float32(m3 / 255.).transpose([2, 0, 1])[::-1]
        m3 = (img_x[index] + m3)
        m3 = m3 / m3.max()

        vis.images(img_x[index],
                   win='cam_adv_reg_%d' % index,
                   opts=dict(title='reg_cam_reg_%d, true label:, target label:' % (index)))
        vis.images(m3,
                   win='cam_adv_reg_map_%d' % index,
                   opts=dict(title='reg_cam_reg_img_%d_resnet, true label' % (index)))

        # vis.images(imgs,
        #            win='img_adv_%d' % index,
        #            opts=dict(title='acid_cam_adv_img_%d_resnet, true label' % (index)))
        # vis.images(m2,
        #            win='img_%d' % index,
        #            opts=dict(title='acid_cam_map_img_%d_resnet, true label: ' % (index)))
        # vis.images(np.concatenate([np.stack(imgs), np.stack(cams)], axis=0), nrow=len(imgs) // 2,
        #            win='img_%d' % index, opts=dict(title='img %d' % index))
        # vis.text('y: %s, yt: %s' % (imagenet_labels[img_y[index]][1], imagenet_labels[img_yt[index]][1]),
        #          win='info_%d' % index, opts=dict(title='info %d' % index))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('data_path')
    # parser.add_argument('data_adv_path')
    parser.add_argument('cam_benign_path')
    parser.add_argument('cam_adv_path')
    config = parser.parse_args()

    print('Please check the configuration', config)
    main(config)
