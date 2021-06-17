#!/usr/bin/env python
import argparse

import numpy as np
from PIL import Image
import visdom

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_paths", metavar="IMAGES", nargs="+")
    parser.add_argument("-n", "--name", dest="name")
    parser.add_argument("-e", "--env", dest="env")

    flags = parser.parse_args()
    paths = flags.image_paths

    vis = visdom.Visdom()
    for path in paths:
        print(path)
        img = Image.open(path)
        vis.image(np.transpose(np.array(img), [2, 0, 1]), win=flags.name, env=flags.env)

