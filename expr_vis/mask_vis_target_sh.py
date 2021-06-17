

# CUDA_VISIBLE_DEVICES=0 python expr_vis/vis_mask.py /home/xinyang/Data/intattack/rev1/target_maps/mask_v2_densenet169/fold_1.npz /home/ningfei/xinyang/data_mask/target_mask_densenet/fold_1.npz

# CUDA_VISIBLE_DEVICES=0 python expr_vis/mask_vis_target_sh.py /home/xinyang/Data/intattack/rev1/random_shape/bmask/fold_1.npz

import argparse
import os
import numpy as np
import cv2
import visdom
import torch
import torch.nn.functional as F

from ia_utils.data_utils import load_imagenet_labels

vis = visdom.Visdom(env='mask_target_shape', port=7777)


def resize_torch_image(x, new_size=(224, 224)):
    try:
        x = x.numpy()
    except Exception as ex:
        pass
    x = x.transpose([1, 2, 0])
    x = cv2.resize(x, new_size)
    x = x.transpose([2, 0, 1])[::-1].copy()
    return x


def transform_mask(img, mask, new_size=(224, 224)):
    m1 = np.uint8(255 * cv2.resize(1 - mask[0], new_size, interpolation=cv2.INTER_LINEAR))
    m1 = cv2.applyColorMap(m1, cv2.COLORMAP_JET)
    m1 = np.float32(m1 / 255.).transpose([2, 0, 1])[::-1]
    m1 = (img + m1)
    m1 = m1 / m1.max()
    return m1


def main(config):
    # dobj = np.load(config.data_path)
    # dobj_adv = np.load(config.data_adv_path)
    # dobj_mask_benign = np.load(config.mask_benign_path)
    dobj_mask_adv = np.load(config.mask_adv_path)
    img_x = dobj_mask_adv['img_x'].copy()
    adv_x = dobj_mask_adv['adv_x'].copy()
    adv_mask = dobj_mask_adv['adv_mask'].copy()
    tar_mask = dobj_mask_adv['tmask'].copy()
    succ = dobj_mask_adv['adv_succeed'].copy()
    benign_map = dobj_mask_adv['bmask'].copy()
    # img_x, img_y, img_yt = dobj_mask_adv['img_x'].copy(), dobj_mask_adv['img_y'].copy(), dobj_mask_adv['img_yt'].copy()
    # print(list(dobj_mask_adv.keys()))
    # mask_benign_y = dobj_mask_adv['mask_benign_y'].copy()
    # # mask_benign_yt = dobj_mask_benign['mask_benign_yt'].copy()
    # indices = np.random.RandomState(100).choice(len(img_x), size=100, replace=False)
    # tar_img = dobj_mask_adv['mask_benign_y'].copy()


    # mask_benign_y = dobj_mask_benign['mask_benign_y'].copy()
    # mask_benign_yt = [dobj_mask_benign['pgd_step_%d_adv_mask_yt' % vis_step].copy() for vis_step in vis_steps]

    # img_x, img_y, img_yt = arrs_masks['img_x'].copy(), dobj['img_y'].copy(), dobj['img_yt'].copy()
    bad_image = torch.tensor(np.zeros((3, 224, 224), dtype=np.float32))

    imagenet_labels = load_imagenet_labels()

    for i in range(30):
        # index = indices[i]
        # vis.image(img_x[i], win='img_%d' % index, opts=dict(title='img_%d' % index))
        # y, yt = img_y[i], img_yt[i]
        img, flag = img_x[i],succ[i]
        if flag == 0:
            p = 0
                # img.append(bad_image)
                # vis.images(bad_image,
                #         win='acid_pgd_adv_%d' % i,
                #         opts=dict(title='acid_pgd_adv_%d,  true label: %s, target label: %s' % (i, imagenet_labels[y][1], imagenet_labels[yt][1])))
                # vis.images(bad_image,
                #         win='acid_map_pgd_adv_%d' % i,
                #         opts=dict(title='acid_map_pgd_adv_%d,  true label: %s, target label: %s' % (i, imagenet_labels[y][1], imagenet_labels[yt][1])))
        else:

        # m1 = transform_mask(img_x[index], mask_benign_y[index])
        # vis.images([resize_torch_image(img_x[index]), resize_torch_image(m1)], win='mask_y_%d' % index, opts=dict(title='mask_y_%d' % index))
                m1 = np.uint8(255 * cv2.resize(1 - benign_map[i,0], (224, 224), interpolation=cv2.INTER_LINEAR))
                m1 = cv2.applyColorMap(m1, cv2.COLORMAP_JET)
                m1 = np.float32(m1 / 255.).transpose([2, 0, 1])[::-1]
                # print(np.shape(img[i]))
                m1 = (img + m1)
                m1 = m1 / m1.max()
                vis.images(img,
                           win='benign_%d' % i,
                           opts=dict(title='benign_%d' % (i)))
                vis.images(m1,
                           win='benign_map_%d' % i,
                           opts=dict(title='benign_map_%d' % (i)))

                m3 = np.uint8(255 * cv2.resize(1 - tar_mask[i,0], (224, 224), interpolation=cv2.INTER_LINEAR))
                m3 = cv2.applyColorMap(m3, cv2.COLORMAP_JET)
                m3 = np.float32(m3 / 255.).transpose([2, 0, 1])[::-1]
                m3 = (img + m3)
                m3 = m3 / m3.max()
                # vis.images(img,
                #            win='benign_%d' % i,
                #            opts=dict(title='benign_%d' % (i)))
                vis.images(m3,
                           win='target_map_%d' % i,
                           opts=dict(title='target_map_%d' % (i)))

                m1 = np.uint8(255 * cv2.resize(1 - adv_mask[i, 0], (224, 224), interpolation=cv2.INTER_LINEAR))
                m1 = cv2.applyColorMap(m1, cv2.COLORMAP_JET)
                m1 = np.float32(m1 / 255.).transpose([2, 0, 1])[::-1]
                m1 = (adv_x[i] + m1)
                m1 = m1 / m1.max()
                vis.images(adv_x[i],
                           win='acid_%d' % i,
                           opts=dict(title='acid_%d' % (i)))
                vis.images(m1,
                           win='acid_map_%d' % i,
                           opts=dict(title='acid_map_%d' % (i)))

        # to_show = []
        # to_show_mask = []
        # for j in range(len(arrs_imgs)):
        #     img, flag = arrs_imgs[j][index], arrs_flags[j][index]
        #     if flag == 0:
        #         to_show.append(bad_image)
        #         to_show_mask.append(bad_image)
        #     else:
        #         img = cv2.putText(np.uint8(255 * img[::-1].transpose([1, 2, 0])).copy(), 'step %d' % vis_steps[j],
        #                     (10, 100),
        #                     cv2.FONT_HERSHEY_SIMPLEX,
        #                     1,
        #                     (255, 255, 255),
        #                     2)
        #         img = cv2.resize(img, (224, 224))
        #         img = np.float32(img.transpose([2, 0, 1])[::-1].copy() / 255.)
        #         to_show.append(torch.tensor(img))
        #         m2 = transform_mask(img, arrs_masks[j][index], (224, 224))
        #         to_show_mask.append(torch.tensor(m2))
        # index = indices[i]
        # img, y, yt, path = img_x[index], img_y[index], img_yt[index], img_path[index]
        # # vis.image(img_x[index], win='img_%d' % index, opts=dict(title='img_%d' % index))
        # m1 = np.uint8(255 * cv2.resize(1 - mask_benign_y[index, 0], (224, 224), interpolation=cv2.INTER_LINEAR))
        # m1 = cv2.applyColorMap(m1, cv2.COLORMAP_JET)
        # m1 = np.float32(m1 / 255.).transpose([2, 0, 1])[::-1]
        # m1 = (img_x[index] + m1)
        # m1 = m1 / m1.max()


                # vis.images(img,
                #         win='acid_pgd_adv_%d' % i,
                #         opts=dict(title='acid_pgd_adv_%d,  true label: %s, target label: %s' % (i, imagenet_labels[y][1], imagenet_labels[yt][1])))
                # vis.images(m1,
                #         win='acid_map_pgd_adv_%d' % i,
                #         opts=dict(title='acid_map_pgd_adv_%d,  true label: %s, target label: %s' % (i, imagenet_labels[y][1], imagenet_labels[yt][1])))
        # vis.text('true label: %s, target label: %s' % (imagenet_labels[img_y[index]],
        #                                                imagenet_labels[img_yt[index]]),
        #          win='info_%d' % index, opts=dict(title='info_%d' % index))
        # to_show = torch.stack(to_show)
        # to_show_mask = torch.stack(to_show_mask)
        # vis.images(to_show, nrow=3, win='adv_%d' % index, opts=dict(title='adv_%d' % index))
        # vis.images(to_show_mask, nrow=3, win='adv_mask_%d' % index, opts=dict(title='adv_mask_%d' % index))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('data_path')
    # parser.add_argument('data_adv_path')
    # parser.add_argument('mask_benign_path')
    parser.add_argument('mask_adv_path')
    config = parser.parse_args()

    print('Please check the configuration', config)
    main(config)
