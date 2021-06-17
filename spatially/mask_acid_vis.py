# CUDA_VISIBLE_DEVICES=0 python expr_vis/vis_mask.py /home/xinyang/Data/intattack/rev1/target_maps/mask_v2_resnet50/fold_1.npz /home/ningfei/xinyang/data_mask/target_mask/fold_1.npz
# CUDA_VISIBLE_DEVICES=0 python expr_vis/vis_mask.py /home/ningfei/xinyang/data_mask/mask_vis_resnet/benign_mask_fold_1.npz

# CUDA_VISIBLE_DEVICES=0 python expr_vis/vis_mask.py /home/ningfei/xinyang/data_mask/mask_vis_densenet/fold_1.npz


# CUDA_VISIBLE_DEVICES=0 python rev1/mask_v2/mask_attack_acid_alt.py /home/xinyang/Data/intattack/fixup1/densenet_data/fold_1.npz /home/xinyang/Data/intattack/rev1/benign_maps/mask_v2_densenet169_benign/fold_1.npz densenet169 ../data_mask/acid_densenet_mask/fold_1.npz -b 20

# CUDA_VISIBLE_DEVICES=0 python rev1/mask_v2/mask_attack_acid_alt.py /home/xinyang/Data/intattack/fixup1/densenet_data/fold_1.npz /home/xinyang/Data/intattack/rev1/benign_maps/mask_v2_densenet169_benign/fold_1.npz densenet169 ../data_mask/acid_densenet_mask/fold_1.npz -b 34

# CUDA_VISIBLE_DEVICES=0 python spatially/mask_acid_vis.py /home/xinyang/Data/intattack/rev1/acid_stadv_maps/mask_densenet169_benign/fold_2.npz /home/xinyang/Data/intattack/rev1/regular_stadv_maps/mask_v2_densenet169/fold_2.npz

import argparse
import os
import numpy as np
import cv2
import visdom
import torch
import torch.nn.functional as F

from ia_utils.data_utils import load_imagenet_labels

vis = visdom.Visdom(env='st_mask_reg_den', port=7778)


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
    dobj_mask_benign = np.load(config.mask_benign_path)
    dobj_mask_adv = np.load(config.mask_adv_path)
    img_x, img_y, img_yt = dobj_mask_benign['img_x'].copy(), dobj_mask_benign['img_y'].copy(), dobj_mask_benign['img_yt'].copy()
    print(list(dobj_mask_adv.keys()))
    mask_benign_y = dobj_mask_adv['stadv_mask_yt'].copy()
    # mask_benign_yt = dobj_mask_benign['mask_benign_yt'].copy()
    indices = np.random.RandomState(100).choice(len(img_x), size=100, replace=False)
    # tar_img = dobj_mask_adv['mask_benign_y'].copy()

    # vis_steps = [1000]
    # arrs_imgs = [dobj_mask_adv['pgd_s2_step_%d_adv_x_disc' % vis_step].copy() for vis_step in vis_steps]
    # arrs_flags = [dobj_mask_adv['pgd_s2_step_%d_succeed_disc' % vis_step].copy() for vis_step in vis_steps]
    # arrs_masks = [dobj_mask_adv['pgd_s2_step_%d_adv_mask_m' % vis_step].copy() for vis_step in vis_steps]

    # mask_benign_y = dobj_mask_benign['mask_benign_y'].copy()
    # mask_benign_yt = [dobj_mask_benign['pgd_step_%d_adv_mask_yt' % vis_step].copy() for vis_step in vis_steps]

    # img_x, img_y, img_yt = arrs_masks['img_x'].copy(), dobj['img_y'].copy(), dobj['img_yt'].copy()
    bad_image = torch.tensor(np.zeros((3, 224, 224), dtype=np.float32))

    imagenet_labels = load_imagenet_labels()

    for i in range(100):
        # index = indices[i]
        # vis.image(img_x[i], win='img_%d' % index, opts=dict(title='img_%d' % index))
        y, yt = img_y[i], img_yt[i]
        # img, flag = arrs_imgs[0][i], arrs_flags[0][i]
        # if flag == 0:
        #         # img.append(bad_image)
        #         vis.images(bad_image,
        #                 win='acid_pgd_adv_%d' % i,
        #                 opts=dict(title='acid_pgd_adv_%d,  true label: %s, target label: %s' % (i, imagenet_labels[y][1], imagenet_labels[yt][1])))
        #         vis.images(bad_image,
        #                 win='acid_map_pgd_adv_%d' % i,
        #                 opts=dict(title='acid_map_pgd_adv_%d,  true label: %s, target label: %s' % (i, imagenet_labels[y][1], imagenet_labels[yt][1])))
        # else:
        #
        # # m1 = transform_mask(img_x[index], mask_benign_y[index])
        # # vis.images([resize_torch_image(img_x[index]), resize_torch_image(m1)], win='mask_y_%d' % index, opts=dict(title='mask_y_%d' % index))
        #         m1 = np.uint8(255 * cv2.resize(1 - arrs_masks[0][i,0], (224, 224), interpolation=cv2.INTER_LINEAR))
        #         m1 = cv2.applyColorMap(m1, cv2.COLORMAP_JET)
        #         m1 = np.float32(m1 / 255.).transpose([2, 0, 1])[::-1]
        #         m1 = (img_x[i] + m1)
        #         m1 = m1 / m1.max()
        # m1 = transform_mask(img_x[index],arrs_masks[0][index])
        # vis.image(m2, win='mask_yt_%d' % index, opts=dict(title='mask_yt_%d' % index))
        m3 = np.uint8(255 * cv2.resize(1 - mask_benign_y[i,0], (224, 224), interpolation=cv2.INTER_LINEAR))
        m3 = cv2.applyColorMap(m3, cv2.COLORMAP_JET)
        m3 = np.float32(m3 / 255.).transpose([2, 0, 1])[::-1]
        m3 = (img_x[i] + m3)
        m3 = m3 / m3.max()
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

        vis.images(img_x[i],
                   win='acid_mask_adv_tar_%d' % i,
                   opts=dict(title='acid_mask_tar_%d, true label: %s, target label: %s' % (i, imagenet_labels[y][1], imagenet_labels[yt][1])))
        vis.images(m3,
                   win='acid_mask_adv_tar_map_%d' % i,
                   opts=dict(title='acid_mask_adv_tar_map_%d,  true label: %s, target label: %s' % (i, imagenet_labels[y][1], imagenet_labels[yt][1])))
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
    parser.add_argument('mask_benign_path')
    parser.add_argument('mask_adv_path')
    config = parser.parse_args()

    print('Please check the configuration', config)
    main(config)
