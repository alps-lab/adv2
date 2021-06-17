import argparse

import numpy as np
import cv2
import visdom
import torch
import torch.nn.functional as F

from ia_utils.data_utils import load_imagenet_labels

vis = visdom.Visdom(env='exprmaskadvvis', port=7778)


def resize_torch_image(x, new_size=(112, 112)):
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
    dobj = np.load(config.data_path)
    dobj_adv = np.load(config.data_adv_path)
    dobj_mask_benign = np.load(config.mask_benign_path)
    dobj_mask_adv = np.load(config.mask_adv_path)
    img_x, img_y, img_yt = dobj['img_x'].copy(), dobj['img_y'].copy(), dobj['img_yt'].copy()

    mask_benign_y = dobj_mask_benign['mask_benign_y'].copy()
    mask_benign_yt = dobj_mask_benign['mask_benign_yt'].copy()
    indices = np.random.RandomState(100).choice(len(img_x), size=40, replace=False)

    vis_steps = [50, 100, 150, 200, 250, 300]
    arrs_imgs = [dobj_adv['pgd_step_%d_adv_x' % vis_step].copy() for vis_step in vis_steps]
    arrs_flags = [dobj_adv['pgd_step_%d_succeed' % vis_step].copy() for vis_step in vis_steps]
    arrs_masks = [dobj_mask_adv['pgd_step_%d_adv_mask_yt' % vis_step].copy() for vis_step in vis_steps]
    bad_image = torch.tensor(np.zeros((3, 112, 112), dtype=np.float32))

    imagenet_labels = load_imagenet_labels()

    for i in range(20):
        index = indices[i]
        # vis.image(img_x[i], win='img_%d' % index, opts=dict(title='img_%d' % index))
        m1 = transform_mask(img_x[index], mask_benign_y[index])
        vis.images([resize_torch_image(img_x[index]), resize_torch_image(m1)], win='mask_y_%d' % index, opts=dict(title='mask_y_%d' % index))
        # m2 = transform_mask(img_x[index], mask_benign_yt[index])
        # vis.image(m2, win='mask_yt_%d' % index, opts=dict(title='mask_yt_%d' % index))

        to_show = []
        to_show_mask = []
        for j in range(len(arrs_imgs)):
            img, flag = arrs_imgs[j][index], arrs_flags[j][index]
            if flag == 0:
                to_show.append(bad_image)
                to_show_mask.append(bad_image)
            else:
                img = cv2.putText(np.uint8(255 * img[::-1].transpose([1, 2, 0])).copy(), 'step %d' % vis_steps[j],
                            (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2)
                img = cv2.resize(img, (112, 112))
                img = np.float32(img.transpose([2, 0, 1])[::-1].copy() / 255.)
                to_show.append(torch.tensor(img))
                m2 = transform_mask(img, arrs_masks[j][index], (112, 112))
                to_show_mask.append(torch.tensor(m2))
        vis.text('true label: %s, target label: %s' % (imagenet_labels[img_y[index]],
                                                       imagenet_labels[img_yt[index]]),
                 win='info_%d' % index, opts=dict(title='info_%d' % index))
        to_show = torch.stack(to_show)
        to_show_mask = torch.stack(to_show_mask)
        vis.images(to_show, nrow=3, win='adv_%d' % index, opts=dict(title='adv_%d' % index))
        vis.images(to_show_mask, nrow=3, win='adv_mask_%d' % index, opts=dict(title='adv_mask_%d' % index))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('data_adv_path')
    parser.add_argument('mask_benign_path')
    parser.add_argument('mask_adv_path')
    config = parser.parse_args()

    print('Please check the configuration', config)
    main(config)
