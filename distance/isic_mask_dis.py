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

path_val = ['/home/xinyang/Data/intattack/rev1/benign_maps/isic_mask_benign/fold_1.npz']

path_tar = []
#
path_acid = ['/home/ningfei/xinyang//data_mask/isic_mask.npz']
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

# for k in tar[0].iterkeys():
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
    best_step = []

    for j in range(len(acid[i]['pgd_s2_step_1000_adv_mask_mnow_disc'])):
        distance = 99999999999.0
        best = 0
        for step in range(50,1050,50):
            if acid[i]['pgd_s2_step_%d_succeed_disc' % step][j] == 1:
                tmp_mask_tar = val[i]['mask_benign_y'][j].copy()
                tmp_mask_acid = acid[i]['pgd_s2_step_%d_adv_mask_mnow_disc' % step][j].copy()
                tmp_dis = np.linalg.norm(tmp_mask_tar.reshape((len(tmp_mask_tar),-1)) - tmp_mask_acid.reshape((len(tmp_mask_acid),-1)), 2, axis=1)
                if distance > tmp_dis[0]:
                    distance = tmp_dis[0]
                    best = step
        # print(distance)
        best_step.append(best)
    # print(best_step)
    for j in range(len(acid[i]['pgd_s2_step_400_adv_mask_mnow_disc'])):
        # for s in best_step:
        c2 += 1
        if best_step[j] != 0:

            if acid[i]['pgd_s2_step_%d_succeed_disc' % best_step[j]][j] == 1:
                succeed_acid.append(acid[i]['pgd_s2_step_%d_succeed_disc' % best_step[j]][j].copy())
                con_acid.append(pro(acid[i]['pgd_s2_step_%d_adv_logits' % best_step[j]][j].copy()))
                b_a.append(val[i]['mask_benign_y'][j].copy())
                a.append(acid[i]['pgd_s2_step_%d_adv_mask_mnow_disc' % best_step[j]][j].copy())
                # tt.append(tar[i]['mask_benign_y'][j].copy())
# for i in range(len(path_val)):
#     # for j in range(50, 450, 50)
#     for j in range(len(acid[i]['pgd_s2_step_1000_adv_mask_mnow_disc'])):
#         c2 += 1
#         if acid[i]['pgd_s2_step_1000_succeed'][j] == 1:
#             succeed_acid.append(acid[i]['pgd_s2_step_1000_succeed'][j].copy())
# #             pgd_s2_step_300_adv_logits
#             con_acid.append(pro(acid[i]['pgd_s2_step_1000_adv_logits'][j].copy()))
#             b_a.append(val[i]['mask_benign_y'][j].copy())
#             a.append(acid[i]['pgd_s2_step_400_adv_mask_mnow_disc'][j].copy())
#             # tt.append(tar[i]['mask_benign_yt'][j].copy())
#
print('successful acid: ', float(len(succeed_acid))/float(c2), 'confidence: ', np.array(con_acid).mean())
print(len(succeed_acid))

mask_acid_map = a.copy()
mask_val_map = b_a.copy()
# mask_target_map = tt.copy()

n = len(mask_val_map)
mask_target_map = np.array(mask_val_map)
mask_acid_map = np.array(mask_acid_map)
# print(mask_target_map.shape)
tar_mask_l1_dist = np.linalg.norm(mask_acid_map.reshape((n, -1)) - mask_target_map.reshape((n, -1)), 1, axis=1)
tar_mask_l2_dist = np.linalg.norm(mask_acid_map.reshape((n, -1)) - mask_target_map.reshape((n, -1)), 2, axis=1)

for i in range(len(tar_mask_l1_dist)):
    tar_mask_l1_dist[i] = tar_mask_l1_dist[i]/(mask_target_map.shape[2] * mask_target_map.shape[3])
    tar_mask_l2_dist[i] = tar_mask_l2_dist[i]/math.sqrt((mask_target_map.shape[2] * mask_target_map.shape[3]))
print('tar_acid_mask_l1_dist:', np.mean(tar_mask_l1_dist))
print('tar_acid_mask_l1_dist_std:', np.std(tar_mask_l1_dist))
print('tar_acid_mask_l2_dist:', np.mean(tar_mask_l2_dist))
print('tar_acid_mask_l2_dist_std:', np.std(tar_mask_l2_dist))
print()
#
# n = len(mask_target_map)
# mask_target_map = np.array(mask_target_map)
# mask_val_map = np.array(mask_val_map)
# # print(a.shape)
#
# tar_val_l1_dist = np.linalg.norm(mask_val_map.reshape((n, -1)) - mask_target_map.reshape((n, -1)), 1, axis=1)
# tar_val_l2_dist = np.linalg.norm(mask_val_map.reshape((n, -1)) - mask_target_map.reshape((n, -1)), 2, axis=1)
#
# for i in range(len(tar_val_l1_dist)):
#     tar_val_l1_dist[i] = tar_val_l1_dist[i]/(mask_target_map.shape[2] * mask_target_map.shape[3])
#     tar_val_l2_dist[i] = tar_val_l2_dist[i]/math.sqrt((mask_target_map.shape[2] * mask_target_map.shape[3]))
# print('tar_val_mask_l1_dist:', np.mean(tar_val_l1_dist))
# print('tar_val_mask_l1_dist_std:', np.std(tar_val_l1_dist))
# print('tar_val_mask_l2_dist:', np.mean(tar_val_l2_dist))
# print('tar_val_mask_l2_dist_std:', np.std(tar_val_l2_dist))

