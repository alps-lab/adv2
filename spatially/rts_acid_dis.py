import os
import re
import argparse
import math
import numpy as np
import cv2
import torch


def pro(logit):
    x = torch.tensor(logit)
    b = torch.nn.functional.softmax(x).max()
    return b.numpy()

path_val = ['/home/xinyang/Data/intattack/rev1/acid_stadv_maps/rts_resnet50_benign/fold_1.npz','/home/xinyang/Data/intattack/rev1/acid_stadv_maps/rts_resnet50_benign/fold_2.npz']

# path_tar = ['/home/xinyang/Data/intattack/rev1/target_maps/cam_resnet50/fold_1.npz','/home/xinyang/Data/intattack/rev1/target_maps/cam_resnet50/fold_2.npz']

# path_acid = ['/home/xinyang/Data/intattack/rev1/acid_stadv_maps/rts_densenet169_benign/fold_2.npz','/home/xinyang/Data/intattack/rev1/acid_stadv_maps/rts_densenet169_benign/fold_3.npz']
path_acid = ['/home/xinyang/Data/intattack/rev1/acid_stadv_maps/rts_densenet169_benign/fold_1.npz','/home/xinyang/Data/intattack/rev1/acid_stadv_maps/rts_densenet169_benign/fold_2.npz']

tar = []
acid = []
val = []
reg = []
d = []
for i in range(len(path_val)):
    acid.append(np.load(path_acid[i]))
    val.append(np.load(path_val[i]))
    # tar.append(np.load(path_tar[i]))

for k in acid[0].iterkeys():
    print(k)

succeed_reg = []
succeed_acid = []
con_reg = []
con_acid = []
c1 = 0
c2 = 0
b_a = []
b_r = []
a = []
r = []
tt = []
for i in range(len(path_val)):
    for j in range(len(acid[i]['s2_step_600_stadv_rts'])):
        c2 += 1
        # if acid[i]['pgd_s2_step_1200_adv_succeed'][j] == 1:
        #     succeed_acid.append(acid[i]['pgd_s2_step_1200_adv_succeed'][j].copy())
#             pgd_s2_step_300_adv_logits
#         con_acid.append(pro(acid[i]['pgd_s2_step_1200_adv_logits'][j].copy()))
        b_a.append(acid[i]['target_rts'][j].copy())
        a.append(acid[i]['s2_step_600_stadv_rts'][j].copy())
        # tt.append(tar[i]['target_cam_n'][j].copy())

# print('successful acid: ', float(len(succeed_acid))/float(c2), 'confidence: ', np.array(con_acid).mean())
# print(len(succeed_acid))

cam_acid_map = a.copy()
cam_val_map = b_a.copy()
# cam_target_map = tt.copy()

n = len(cam_acid_map)
cam_target_map = np.array(cam_val_map)
cam_acid_map = np.array(cam_acid_map)
# print(cam_target_map.shape)
tar_cam_l1_dist = np.linalg.norm(cam_acid_map.reshape((n, -1)) - cam_target_map.reshape((n, -1)), 1, axis=1)
tar_cam_l2_dist = np.linalg.norm(cam_acid_map.reshape((n, -1)) - cam_target_map.reshape((n, -1)), 2, axis=1)

for i in range(len(tar_cam_l1_dist)):
    tar_cam_l1_dist[i] = tar_cam_l1_dist[i]/(cam_target_map.shape[2] * cam_target_map.shape[3])
    tar_cam_l2_dist[i] = tar_cam_l2_dist[i]/math.sqrt((cam_target_map.shape[2] * cam_target_map.shape[3]))
print('tar_acid_cam_l1_dist:', np.mean(tar_cam_l1_dist))
print('tar_acid_cam_l1_dist_std:', np.std(tar_cam_l1_dist))
print('tar_acid_cam_l2_dist:', np.mean(tar_cam_l2_dist))
print('tar_acid_cam_l2_dist_std:', np.std(tar_cam_l2_dist))
print()
