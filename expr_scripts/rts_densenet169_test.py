#!/usr/bin/env python
import numpy as np
import visdom
import torch
import torch.nn.functional as F

from expr_attacks.rts_model_densenet169 import RTSDensenet169
from expr_attacks.rts_generator import get_default_rts_config, generate_rts_per_batch


if __name__ == '__main__':
    rts_config = get_default_rts_config('densenet169')
    rts_model = RTSDensenet169(rts_config['ckpt_dir'], True)

    resnet_arx = np.load('/home/ningfei/data/data_resnet/fold_2.npz')
    densenet_arx = np.load('/home/ningfei/data/densenet_data/fold_2.npz')

    resnet_img_x, resnet_img_y = resnet_arx['img_x'], resnet_arx['img_y']
    densenet_img_x, densenet_img_y = densenet_arx['img_x'], densenet_arx['img_y']

    batch_size = 30
    n = len(resnet_img_x)
    n_batches = (n + batch_size - 1) // batch_size
    for i in range(n_batches):
        si = i * batch_size
        ei = min(si + batch_size, n)
        bx_res, by_res = resnet_img_x[si:ei], resnet_img_y[si:ei]
        bx_des, by_des = densenet_img_x[si:ei], densenet_img_y[si:ei]

        bx = torch.tensor(bx_res)
        bx = bx.cuda()
        logits = rts_model.logits_fn(bx)
        preds = logits.argmax(1).detach().cpu().numpy()
        print('resnet50 encoder', np.sum((preds == by_res)))

        bx = torch.tensor(bx_des)
        bx = bx.cuda()
        logits = rts_model.blackbox_logits_fn(bx)
        preds = logits.argmax(1).detach().cpu().numpy()
        print('densenet blackbox function', np.sum((preds == by_des)))

        if i == n_batches - 1:
            vis = visdom.Visdom(port=3214, env='xytestrts')
            rts = generate_rts_per_batch(rts_config, rts_model, (bx_des, by_des), True)
            rts_upsampled = F.upsample(rts, (224, 224), mode='bilinear')
            print(rts_upsampled.shape)
            stacked = torch.cat([bx, rts_upsampled.expand(-1, 3, -1, -1)], 0)
            vis.images(stacked, nrow=5)
