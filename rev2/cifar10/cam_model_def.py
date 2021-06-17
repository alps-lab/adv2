import torch
import torch.nn as nn

from rev2.cifar10.model_utils import resnet50, CIFAR10_RESNET50_CKPT_PATH
from rev2.cifar10.data_utils import cifar10_normalize


def cam_resnet50_forward(model_tup, x):
    model, pre_fn = model_tup[:2]
    res = model(pre_fn(x), out_keys=["l4", "gvp", "fc"])
    return res['l4'], res['gvp'], res['fc']


def cam_resnet50_fc_weight(model_tup):
    model = model_tup[0]
    return model.linear.weight


def cam_resnet50():
    model = resnet50()
    model_tup = (model, cifar10_normalize, (32, 32))

    ckpt_dict = torch.load(CIFAR10_RESNET50_CKPT_PATH, lambda storage, location: storage)['net']
    nn.DataParallel(model).load_state_dict(ckpt_dict)

    return model_tup, (cam_resnet50_forward, cam_resnet50_fc_weight)
