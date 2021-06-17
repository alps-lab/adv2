import torch
import numpy as np


def get_default_gradcam_config():
    return dict(batch_size=40)


def generate_gradcam_per_batch(gradcam_model, batch_tup, cuda):
    bx, by = batch_tup
    bx, by = torch.tensor(bx), torch.tensor(by)
    if cuda:
        bx, by = bx.cuda(), by.cuda()
    return gradcam_model(bx, by, False)


def generate_gradcams(gradcam_config, gradcam_model, images_tup, cuda):
    img_x, img_y = images_tup[:2]
    batch_size = gradcam_config['batch_size']
    num_batches = (len(img_x) + batch_size - 1) // batch_size

    gradcams, gradcams_n = [], []
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min(len(img_x), start_index + batch_size)
        bx, by = img_x[start_index:end_index], img_y[start_index:end_index]
        bgradcam, bgradcam_n = generate_gradcam_per_batch(gradcam_model, (bx, by), cuda)[-2:]
        bgradcam = bgradcam.detach().cpu().numpy()
        bgradcam_n = bgradcam_n.detach().cpu().numpy()
        gradcams.append(bgradcam)
        gradcams_n.append(bgradcam_n)

    gradcams = np.concatenate(gradcams, axis=0)
    gradcams_n = np.concatenate(gradcams_n, axis=0)

    return gradcams, gradcams_n

