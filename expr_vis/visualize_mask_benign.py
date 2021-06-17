import argparse

import numpy as np
import cv2
import visdom


vis = visdom.Visdom(env='mask_resnet_benign',port=7778)


def main(config):
    # dobj = np.load(config.data_path)
    dobj_mask = np.load(config.mask_path)
    img_x, img_y, img_yt = dobj_mask['img_x'].copy(), dobj_mask['img_y'].copy(), dobj_mask['img_yt'].copy()

    mask_benign_y = dobj_mask['mask_benign_y'].copy()
    # mask_benign_yt = dobj_mask['mask_benign_yt'].copy()
    indices = np.random.RandomState(100).choice(len(img_x), size=100, replace=False)

    for i in range(100):
        index = indices[i]
        vis.image(img_x[index], win='img_%d' % index, opts=dict(title='img_%d' % index))
        m1 = np.uint8(255 * cv2.resize(1 - mask_benign_y[index, 0], (224, 224), interpolation=cv2.INTER_LINEAR))
        m1 = cv2.applyColorMap(m1, cv2.COLORMAP_JET)
        m1 = np.float32(m1 / 255.).transpose([2, 0, 1])[::-1]
        m1 = (img_x[index] + m1)
        m1 = m1 / m1.max()
        vis.image(m1, win='mask_y_%d' % index, opts=dict(title='mask_y_%d' % index))
        # m2 = np.uint8(255 * cv2.resize(1 - mask_benign_y[index, 0], (224, 224), interpolation=cv2.INTER_LINEAR))
        # m2 = cv2.applyColorMap(m2, cv2.COLORMAP_JET)
        # m2 = np.float32(m2 / 255.).transpose([2, 0, 1])[::-1]
        # m2 = (img_x[index] + m2)
        # m2 = m2 / m2.max()
        # vis.image(m2, win='mask_yt_%d' % index, opts=dict(title='mask_yt_%d' % index))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('data_path')
    parser.add_argument('mask_path')
    config = parser.parse_args()

    print('Please check the configuration', config)
    main(config)
