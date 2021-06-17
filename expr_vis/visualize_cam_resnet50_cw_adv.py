import argparse

import numpy as np
import cv2
import visdom

from ia_utils.data_utils import load_imagenet_labels


vis = visdom.Visdom(env='exprcamadvviscw')


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


def resize(img, new_size=(112, 112)):
    img = np.uint8(255 * img.transpose([1, 2, 0]))
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
    img = img.transpose([2, 0, 1])
    return np.float32(img / 255.)


def generate_succeed_mask(img_x, img_x_adv, best_scores, eps=0.005):
    n = len(img_x)
    diff = (img_x - img_x_adv).reshape((n, -1))
    tot_l2dist = np.linalg.norm(diff, ord=2, axis=-1)
    num_pixels = img_x.shape[2] * img_x.shape[3]
    normed_l2dist = tot_l2dist / np.sqrt(num_pixels)
    flag = np.logical_and(normed_l2dist < eps, best_scores != -1)
    return flag.astype(np.int64)


def main(config):
    dobj = np.load(config.data_path)
    dobj_adv = np.load(config.data_adv_path)
    dobj_cam_benign = np.load(config.cam_benign_path)
    dobj_cam_adv = np.load(config.cam_adv_path)
    img_x, img_y, img_yt = dobj['img_x'].copy(), dobj['img_y'].copy(), dobj['img_yt'].copy()

    cam_benign_y = dobj_cam_benign['cam_benign_y_n'].copy()
    indices = np.random.RandomState(100).choice(len(img_x), size=60, replace=False)
    imagenet_labels = load_imagenet_labels()

    vis_confs = [0, 5]
    arrs_imgs = [dobj_adv['cw_conf_%d_best_attack' % vis_conf].copy() for vis_conf in vis_confs]
    arrs_best_scores = [dobj_adv['cw_conf_%d_best_score' % vis_conf].copy() for vis_conf in vis_confs]
    arrs_flags = [generate_succeed_mask(img_x, img_adv, best_score) for img_adv, best_score
                  in zip(arrs_imgs, arrs_best_scores)]
    # arrs_flags = [dobj_adv['pgd_step_%d_succeed' % vis_step].copy() for vis_conf in vis_steps]
    arrs_cams = [dobj_cam_adv['cw_conf_%d_adv_cam_yt_n' % vis_conf].copy() for vis_conf in vis_confs]
    bad_image = np.zeros((3, 112, 112), dtype=np.float32)

    for i in range(60):
        index = indices[i]
        imgs, texts, cams = [], [], []
        imgs.append(resize(img_x[index]))
        texts.append('benign')
        cams.append(resize(transform_cam(img_x[index], cam_benign_y[index])))

        for j in range(len(arrs_imgs)):
            img, flag = arrs_imgs[j][index], arrs_flags[j][index]
            if flag == 0:
                imgs.append(bad_image)
                texts.append('')
                cams.append(bad_image)
            else:
                imgs.append(resize(arrs_imgs[j][index]))
                dis = np.linalg.norm((img_x[index] - img).flatten(), 2) / 224.
                texts.append('%d, %.4f' % (vis_confs[j], dis))
                cams.append(resize(transform_cam(arrs_imgs[j][index], arrs_cams[j][index])))

        for j, (img, text) in enumerate(zip(imgs, texts)):
            imgs[j] = draw_text(img, text)
        vis.images(np.concatenate([np.stack(imgs), np.stack(cams)], axis=0), nrow=len(imgs) // 2,
                   win='img_%d' % index, opts=dict(title='img %d' % index))
        # vis.text('y: %s, yt: %s' % (imagenet_labels[img_y[index]][1], imagenet_labels[img_yt[index]][1]),
        #          win='info_%d' % index, opts=dict(title='info %d' % index))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('data_adv_path')
    parser.add_argument('cam_benign_path')
    parser.add_argument('cam_adv_path')
    parser.add_argument('-e', '--epsilon', )
    config = parser.parse_args()

    print('Please check the configuration', config)
    main(config)
