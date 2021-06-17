import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, Bottleneck

from ia_utils.model_utils import freeze_model
from pretrainedmodels.models.torchvision_models import pretrained_settings, load_pretrained, modify_resnets
from ia_utils.model_utils import SoftReLU


class BottleNeckSReLU(Bottleneck):

    def __init__(self, *args, **kwargs):
        super(BottleNeckSReLU, self).__init__(*args, **kwargs)
        self.relu = SoftReLU()


def resnet50_srelu(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BottleNeckSReLU, [3, 4, 6, 3], **kwargs)
    model.relu = SoftReLU()
    return model


def resnet50(num_classes=1000, pretrained='imagenet'):
    """Constructs a ResNet-50 model.
    """
    model = resnet50_srelu(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['resnet50'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def get_isic_model_on_resnet50_srelu(train=False, freeze=True, ckpt_path=None):
    model = resnet50()
    model.last_linear = nn.Linear(model.last_linear.in_features, 7)
    model.train(train)

    if freeze:
        freeze_model(model)
    if ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path, map_location=lambda storage, loc: storage)['state_dict'])
    return model
