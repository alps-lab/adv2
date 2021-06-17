import torch
import torch.nn as nn
import math

from torchvision.models.resnet import Bottleneck

from rev1.isic_acid.isic_utils import ISIC_RESNET50_CKPT_PATH
from ia_utils.model_utils import freeze_model


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, out_keys=None):
        out = {}
        x = self.conv1(x)
        out["c1"] = x
        x = self.bn1(x)
        out["bn1"] = x
        x = self.relu(x)
        out["r1"] = x
        x = self.maxpool(x)
        out["p1"] = x

        x = self.layer1(x)
        out["l1"] = x
        x = self.layer2(x)
        out["l2"] = x
        x = self.layer3(x)
        out["l3"] = x
        x = self.layer4(x)
        out["l4"] = x

        x = self.avgpool(x)
        out["gvp"] = x
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        out["fc"] = x

        if out_keys is None:
            return x

        res = {}
        for key in out_keys:
            res[key] = out[key]
        return res


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def identity(x):
    return x


def cam_resnet50_forward(model_tup, x):
    model, pre_fn = model_tup[:2]
    res = model(pre_fn(x), out_keys=["l4", "gvp", "fc"])
    return res['l4'], res['gvp'], res['fc']


def cam_resnet50_fc_weight(model_tup):
    model = model_tup[0]
    return model.last_linear.weight


def cam_resnet50():
    model = resnet50(num_classes=7)
    model.train(False)
    freeze_model(model)
    model.load_state_dict(torch.load(ISIC_RESNET50_CKPT_PATH)['state_dict'])
    model_tup = (model, identity, (224, 224))
    return model_tup, (cam_resnet50_forward, cam_resnet50_fc_weight)


if __name__ == '__main__':
    tups = cam_resnet50()
