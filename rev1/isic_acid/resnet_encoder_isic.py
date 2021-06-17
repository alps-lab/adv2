import torch
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
from torchvision.models.resnet import ResNet, Bottleneck


class ResNetEncoder(ResNet):
    def forward(self, x):
        s0 = x
        x = self.conv1(s0)
        x = self.bn1(x)
        s1 = self.relu(x)
        x = self.maxpool(s1)

        s2 = self.layer1(x)
        s3 = self.layer2(s2)
        s4 = self.layer3(s3)

        s5 = self.layer4(s4)

        x = self.avgpool(s5)
        sX = x.view(x.size(0), -1)
        sC = self.last_linear(sX)

        return s0, s1, s2, s3, s4, s5, sX, sC


def resnet50encoder(num_classes=7, pretrained=False, **kwargs):
    """Constructs a ResNet-50 encoder that returns all the intermediate feature maps.
    For resnet50 the returned feature maps (for example batch size 5) are:
    (5L, 3L, 224L, 224L)
    (5L, 64L, 112L, 112L)
    (5L, 256L, 56L, 56L)
    (5L, 512L, 28L, 28L)
    (5L, 1024L, 14L, 14L)
    (5L, 2048L, 7L, 7L)
    (5L, 2048L)
    (5L, 1000L)

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetEncoder(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    model.last_linear = nn.Linear(model.fc.in_features, num_classes)
    model.fc = None
    return model
