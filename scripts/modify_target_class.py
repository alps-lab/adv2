#!/usr/bin/env python
import argparse
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('save_path')
    parser.add_argument('-s', '--seed', type=int)

    config = parser.parse_args()
    data_arx = np.load(config.input_path)
    data_dict = dict(data_arx.items())
    img_y = data_dict['img_y']
    img_yt = data_arx['img_yt']
    n = len(img_y)
    img_yt = img_y + np.random.RandomState().randint(1, 1000, size=(n,))
    img_yt = np.mod(img_yt, 1000)
    print(np.stack([img_y, img_yt], axis=1)[:30])
    data_dict['img_yt'] = img_yt
    np.savez(config.save_path, **data_dict)
    assert not np.any(img_y == img_yt)
