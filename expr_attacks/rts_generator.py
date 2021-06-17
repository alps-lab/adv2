import numpy as np
import torch


from expr_attacks.commons import RTS_RESNET50_CKPT_DIR, RTS_DENSENET169_CKPT_DIR


def get_default_rts_config(model):
    if model == 'resnet50':
        return dict(ckpt_dir=RTS_RESNET50_CKPT_DIR, batch_size=40, model_confidence=6)
    if model == 'densenet169':
        return dict(ckpt_dir=RTS_DENSENET169_CKPT_DIR, batch_size=30, model_confidence=6)


def generate_rts_per_batch(rts_config, rts_model, batch_tup, cuda):
    bx, by = batch_tup
    bx, by = torch.tensor(bx), torch.tensor(by)
    if cuda:
        bx, by = bx.cuda(), by.cuda()
    return rts_model.saliency_fn(bx, by, model_confidence=rts_config['model_confidence'],
                                 return_classification_logits=False)


def generate_rts(rts_config, rts_model, images_tup, cuda):
    img_x, img_y = images_tup[:2]
    batch_size = rts_config['batch_size']
    num_batches = (len(img_x) + batch_size - 1) // batch_size

    rts = []
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min(len(img_x), start_index + batch_size)
        bx, by = img_x[start_index:end_index], img_y[start_index:end_index]
        rts.append(generate_rts_per_batch(rts_config, rts_model, (bx, by), cuda).detach().cpu().numpy())

    rts = np.concatenate(rts, axis=0)
    return rts
