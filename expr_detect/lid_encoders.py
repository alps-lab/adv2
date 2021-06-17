import re

import torch
import torch.nn.functional as F
import torchvision
import torch.utils.model_zoo as model_zoo


class DenseNetEncoder(torchvision.models.DenseNet):

    def forward(self, x):
        s0 = x
        x = self.features.conv0(s0)
        x = self.features.norm0(x)
        s1 = self.features.relu0(x)
        x = self.features.pool0(s1)
        x = self.features.denseblock1(x)
        s2 = self.features.transition1[:2](x)
        x = self.features.transition1[2:](s2)
        x = self.features.denseblock2(x)
        s3 = self.features.transition2[:2](x)
        x = self.features.transition2[2:](s3)
        x = self.features.denseblock3(x)
        s4 = self.features.transition3[:2](x)
        x = self.features.transition3[2:](s4)
        x = self.features.denseblock4(x)
        x = self.features.norm5(x)
        s5 = F.relu(x, inplace=True)
        sX = F.avg_pool2d(s5, kernel_size=7, stride=1).view(x.size(0), -1)
        sC = self.classifier(sX)
        return s0, s1, s2, s3, s4, s5, sX, sC


def densenet169encoder(pretrained=False, **kwargs):
    model = DenseNetEncoder(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/densenet169-b2777c0a.pth')
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


class ResNetEncoder(torchvision.models.ResNet):

    def forward(self, x):
        s0 = x
        s1 = self.conv1(s0)
        s2 = self.bn1(s1)
        s3 = self.relu(s2)
        s4 = self.maxpool(s3)

        x = s4
        s5s, s6s, s7s, s8s = [], [], [], []
        for block in self.layer1.children():
            x = block(x)
            s5s.append(x)
        for block in self.layer2.children():
            x = block(x)
            s6s.append(x)
        for block in self.layer3.children():
            x = block(x)
            s7s.append(x)
        for block in self.layer4.children():
            x = block(x)
            s8s.append(x)
        s9 = self.avgpool(x)
        s10 = s9.view(s9.size(0), -1)
        s11 = self.fc(s10)
        l = [s0, s3, s4] + s5s + s6s + s7s + s8s + [s10, s11]
        return l


def resnet50encoder(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetEncoder(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    return model

