import torch
import torch.nn as nn
from pretrainedmodels import resnet50

from ia_utils.model_utils import freeze_model


def get_isic_model_on_resnet50(train=False, freeze=True, ckpt_path=None):
    model = resnet50()
    model.last_linear = nn.Linear(model.last_linear.in_features, 7)
    model.train(train)

    if freeze:
        freeze_model(model)
    if ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path, map_location=lambda storage, loc: storage)['state_dict'])
    return model
