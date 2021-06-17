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

path_val = ['/home/xinyang/Data/intattack/fixup1/densenet169_regular_pgd_rts/fold_1.npz','/home/xinyang/Data/intattack/fixup1/densenet169_regular_pgd_rts/fold_2.npz']

path_tar = ['/home/xinyang/Data/intattack/rev1/target_maps/rts_densenet169/fold_1.npz','/home/xinyang/Data/intattack/rev1/target_maps/rts_densenet169/fold_2.npz']

path_acid = ['/home/ningfei/xinyang/data_fix1/target/rts_densenet169/fold_1.npz', '/home/ningfei/xinyang/data_fix1/target/rts_densenet169/fold_2.npz']

tar = []
acid = []
val = []
reg = []
d = []
for i in range(len(path_val)):
    acid.append(np.load(path_acid[i]))
    val.append(np.load(path_val[i]))
    tar.append(np.load(path_tar[i]))

# for k in val[0].iterkeys():
#     print(k)

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
    for j in range(len(acid[i]['pgd_s2_step_1200_adv_rts'])):
        c2 += 1
        if acid[i]['pgd_s2_step_1200_adv_succeed'][j] == 1:
            succeed_acid.append(acid[i]['pgd_s2_step_1200_adv_succeed'][j].copy())
#             pgd_s2_step_300_adv_logits
            con_acid.append(pro(acid[i]['pgd_s2_step_1200_adv_logits'][j].copy()))
            b_a.append(val[i]['pgd_step_1200_adv_rts_yt'][j].copy())
            a.append(acid[i]['pgd_s2_step_1200_adv_rts'][j].copy())
            tt.append(tar[i]['target_rts'][j].copy())

print('successful acid: ', float(len(succeed_acid))/float(c2), 'confidence: ', np.array(con_acid).mean())
print(len(succeed_acid))

rts_acid_map = a.copy()
rts_val_map = b_a.copy()
rts_target_map = tt.copy()

n = len(rts_target_map)
rts_target_map = np.array(rts_target_map)
rts_acid_map = np.array(rts_acid_map)
# print(rts_target_map.shape)
tar_rts_l1_dist = np.linalg.norm(rts_acid_map.reshape((n, -1)) - rts_target_map.reshape((n, -1)), 1, axis=1)
tar_rts_l2_dist = np.linalg.norm(rts_acid_map.reshape((n, -1)) - rts_target_map.reshape((n, -1)), 2, axis=1)

for i in range(len(tar_rts_l1_dist)):
    tar_rts_l1_dist[i] = tar_rts_l1_dist[i]/(rts_target_map.shape[2] * rts_target_map.shape[3])
    tar_rts_l2_dist[i] = tar_rts_l2_dist[i]/math.sqrt((rts_target_map.shape[2] * rts_target_map.shape[3]))
print('tar_acid_rts_l1_dist:', np.mean(tar_rts_l1_dist))
print('tar_acid_rts_l1_dist_std:', np.std(tar_rts_l1_dist))
print('tar_acid_rts_l2_dist:', np.mean(tar_rts_l2_dist))
print('tar_acid_rts_l2_dist_std:', np.std(tar_rts_l2_dist))
print()

n = len(rts_target_map)
rts_target_map = np.array(rts_target_map)
rts_val_map = np.array(rts_val_map)
# print(a.shape)

tar_val_l1_dist = np.linalg.norm(rts_val_map.reshape((n, -1)) - rts_target_map.reshape((n, -1)), 1, axis=1)
tar_val_l2_dist = np.linalg.norm(rts_val_map.reshape((n, -1)) - rts_target_map.reshape((n, -1)), 2, axis=1)

for i in range(len(tar_val_l1_dist)):
    tar_val_l1_dist[i] = tar_val_l1_dist[i]/(rts_target_map.shape[2] * rts_target_map.shape[3])
    tar_val_l2_dist[i] = tar_val_l2_dist[i]/math.sqrt((rts_target_map.shape[2] * rts_target_map.shape[3]))
print('tar_val_rts_l1_dist:', np.mean(tar_val_l1_dist))
print('tar_val_rts_l1_dist_std:', np.std(tar_val_l1_dist))
print('tar_val_rts_l2_dist:', np.mean(tar_val_l2_dist))
print('tar_val_rts_l2_dist_std:', np.std(tar_val_l2_dist))
