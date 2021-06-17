#!/usr/bin/env python
import numpy as np
import cv2

from shapely.affinity import rotate
from shapely.geometry import box


def decorate_map(target_map, rs):
    hx, hy = np.nonzero(target_map > 0.9)
    lx, ly= np.nonzero(target_map <= 0.9)
    target_map = target_map.copy()
    target_map[hx, hy] = rs.uniform(0.9, 1.0, size=len(hx))
    target_map[lx, ly] = rs.uniform(0, 0.1, size=len(lx))
    return target_map


def create_circle_map(cx, cy, r):
    img = np.zeros((224, 224), np.float32)
    img = cv2.circle(img.copy(), (cx, cy), r, 1, thickness=-1)
    return img


def create_rectangle(cx, cy, hw, hh, angle):
    b = box(cx - hw, cy - hh, cx + hw, cy + hh)
    b = rotate(b, angle)
    pts = np.round(np.stack(b.boundary.xy, 1)).astype(np.int64)
    pts = np.maximum(0, pts)
    pts = np.minimum(223, pts)
    img = np.zeros((224, 224), np.float32)
    img = cv2.drawContours(img.copy(), [pts], 0, 1, thickness=-1)
    return img


def generate_shapes(n=100, seed=777):
    rs = np.random.RandomState(seed)
    shape = np.random.randint(2, size=(n,))  # 0 - rectangle 1 - circle

    target_maps = []
    for i in range(n):
        cx = rs.randint(40, 224 - 40)
        cy = rs.randint(40, 224 - 40)
        if shape[i] == 0:
            r = np.random.randint(30, 60)
            target_maps.append(create_circle_map(cx, cy, r))
        else:
            hw, hh = rs.uniform(30, 60, size=(2, ))
            angle = rs.uniform(-30, 30)
            target_maps.append(create_rectangle(cx, cy, hw, hh, angle))
    target_maps = [decorate_map(target_map, rs) for target_map in target_maps]
    return target_maps
