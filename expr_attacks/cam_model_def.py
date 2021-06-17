import torch
import torch.nn as nn
import torch.nn.functional as F


from ia_utils.model_utils import resnet50, densenet169
from ia_utils.data_utils import imagenet_normalize


def cam_resnet50_forward(model_tup, x):
    model, pre_fn = model_tup[:2]
    res = model(pre_fn(x), out_keys=["l4", "gvp", "fc"])
    return res['l4'], res['gvp'], res['fc']


def cam_resnet50_fc_weight(model_tup):
    model = model_tup[0]
    return model.fc.weight


def cam_resnet50():
    model = resnet50(pretrained=True)
    model_tup = (model, imagenet_normalize, (224, 224))

    return model_tup, (cam_resnet50_forward, cam_resnet50_fc_weight)


def cam_densenet169_forward(model_tup, x):
    model, pre_fn = model_tup[:2]
    res = model(pre_fn(x), out_keys=['l', 'gvp', 'fc'])
    return res['l'], res['gvp'], res['fc']


def cam_densenet169_fc_weight(model_tup):
    model = model_tup[0]
    return model.classifier.weight


def cam_densenet169():
    model = densenet169(pretrained=True)
    model_tup = (model, imagenet_normalize, (224, 224))

    return model_tup, (cam_densenet169_forward, cam_densenet169_fc_weight)
