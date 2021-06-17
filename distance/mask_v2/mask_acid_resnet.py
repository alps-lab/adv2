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

# benign
path_acid = ['/home/ningfei/xinyang/data_mask/reg_pgd_densenet/fold_1.npz']

# /home/xinyang/Data/intattack/rev1/benign_maps/mask_v2_resnet50_benign/fold_1.npz
path_val = ['/home/xinyang/Data/intattack/rev1/benign_maps/mask_v2_densenet169_benign/fold_1.npz']

# path_tar = ['/home/xinyang/Data/intattack/fixup1/resnet50_mask/fold_1.npz','/home/xinyang/Data/intattack/fixup1/resnet50_mask/fold_2.npz']
#
# path_acid = ['/home/xinyang/Data/intattack/rev1/benign_maps/mask_resnet50_alt/fold_1.npz','/home/xinyang/Data/intattack/rev1/benign_maps/mask_resnet50_alt/fold_2.npz']
# ori
# path_acid = ['/home/ningfei/xinyang/data_fix1/target/mask_resnet50/fold_1.npz','/home/ningfei/xinyang/data_fix1/target/mask_resnet50/fold_2.npz']

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
    # for j in range(50, 450, 50)
    for j in range(len(acid[i]['pgd_step_1000_adv_mask_yt'])):
        c2 += 1
        if True:
        # if acid[i]['pgd_s2_step_400_succeed'][j] == 1:
            # succeed_acid.append(acid[i]['pgd_s2_step_400_succeed'][j].copy())
#             pgd_s2_step_300_adv_logits
            con_acid.append(pro(acid[i]['pgd_step_1000_adv_mask_yt'][j].copy()))
            b_a.append(val[i]['mask_benign_y'][j].copy())
            a.append(acid[i]['pgd_step_1000_adv_mask_yt'][j].copy())
            # tt.append(tar[i]['mask_benign_yt'][j].copy())

print('successful acid: ', float(len(succeed_acid))/float(c2), 'confidence: ', np.array(con_acid).mean())
print(len(succeed_acid))

mask_acid_map = a.copy()
mask_val_map = b_a.copy()
# mask_target_map = tt.copy()

n = len(mask_acid_map)
# print(n,"11111")
# print(mask_val_map.shape)
mask_target_map = np.array(mask_acid_map)
mask_val_map = np.array(mask_val_map)
print(mask_val_map.shape)
# print(a.shape)

tar_val_l1_dist = np.linalg.norm(mask_val_map.reshape((n, -1)) - mask_target_map.reshape((n, -1)), 1, axis=1)
tar_val_l2_dist = np.linalg.norm(mask_val_map.reshape((n, -1)) - mask_target_map.reshape((n, -1)), 2, axis=1)

for i in range(len(tar_val_l1_dist)):
    tar_val_l1_dist[i] = tar_val_l1_dist[i]/(mask_target_map.shape[2] * mask_target_map.shape[3])
    tar_val_l2_dist[i] = tar_val_l2_dist[i]/math.sqrt((mask_target_map.shape[2] * mask_target_map.shape[3]))
print('tar_val_mask_l1_dist:', np.mean(tar_val_l1_dist))
print('tar_val_mask_l1_dist_std:', np.std(tar_val_l1_dist))
print('tar_val_mask_l2_dist:', np.mean(tar_val_l2_dist))
print('tar_val_mask_l2_dist_std:', np.std(tar_val_l2_dist))

