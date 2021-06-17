#!/usr/bin/env python
import numpy as np
import torch

from expr_attacks.rts_model import RTS
from expr_attacks.rts_generator import get_default_rts_config


if __name__ == '__main__':
    rts_config = get_default_rts_config()
    rts = RTS(rts_config['ckpt_dir'], True)

    benign_path = '/home/ningfei/data/data_resnet/fold_%d.npz'
    pgd_adv_path = '/home/ningfei/data/data_resnet_2/mask_attack_pgd_vanilla_fold_resnet_%d.npz'

    for i in range(3):
        fold = i + 1

        img_arx = np.load(benign_path % fold)
        img_x, img_y = img_arx['img_x'], img_arx['img_yt']
        pgd_arx = np.load(pgd_adv_path % fold)
        img_adv_x = pgd_arx['pgd_step_2000_adv_x']

        batch_size = 20
        num_batches = (len(img_x) + batch_size - 1) // batch_size
        for j in range(num_batches):
            si = j * batch_size
            ei = min(si + batch_size, len(img_x))
            bx, by = img_adv_x[si:ei], img_y[si:ei]
            logits = rts.logits_fn(torch.tensor(bx).cuda())
            preds = logits.argmax(1).detach().cpu().numpy()
            print('fold %d, batch %d, correct %d, total %d' % (fold, j + 1, np.count_nonzero(preds == by), len(img_x)))
