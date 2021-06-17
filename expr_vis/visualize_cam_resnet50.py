import argparse

import numpy as np
import cv2
import visdom


vis = visdom.Visdom(env='exprcamvis')


def main(config):
    dobj = np.load(config.data_path)
    dobj_mask = np.load(config.cam_path)
    print(dobj_mask.keys())
    img_x, img_y, img_yt = dobj['img_x'].copy(), dobj['img_y'].copy(), dobj['img_yt'].copy()

    cam_benign_y = dobj_mask['cam_benign_y_n'].copy()
    cam_benign_yt = dobj_mask['cam_benign_yt_n'].copy()
    indices = np.random.RandomState(100).choice(len(img_x), size=40, replace=False)

    for i in range(40):
        index = indices[i]
        vis.image(img_x[index], win='img_%d' % index, opts=dict(title='img_%d' % index))
        m1 = np.uint8(255 * cv2.resize(cam_benign_y[index, 0], (224, 224), interpolation=cv2.INTER_LINEAR))
        m1 = cv2.applyColorMap(m1, cv2.COLORMAP_JET)
        m1 = np.float32(m1 / 255.).transpose([2, 0, 1])[::-1]
        m1 = (img_x[index] + m1)
        m1 = m1 / m1.max()
        vis.image(m1, win='cam_y_%d' % index, opts=dict(title='cam_y_%d' % index))
        m2 = np.uint8(255 * cv2.resize(cam_benign_yt[index, 0], (224, 224), interpolation=cv2.INTER_LINEAR))
        m2 = cv2.applyColorMap(m2, cv2.COLORMAP_JET)
        m2 = np.float32(m2 / 255.).transpose([2, 0, 1])[::-1]
        m2 = (img_x[index] + m2)
        m2 = m2 / m2.max()
        vis.image(m2, win='cam_yt_%d' % index, opts=dict(title='cam_yt_%d' % index))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('cam_path')
    config = parser.parse_args()

    print('Please check the configuration', config)
    main(config)
