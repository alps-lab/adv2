from torchvision.models.resnet import resnet50

from ia_utils.data_utils import imagenet_normalize


def gradcam_resnet50_extractor(model_tup, x):
    model, pre_fn = model_tup[:2]
    x = pre_fn(x)
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    features = [x]

    x = model.avgpool(x)
    x = x.view(x.size(0), -1)
    x = model.fc(x)

    return features, x


def gradcam_resnet50():
    model = resnet50(pretrained=True)
    model_tup = (model, imagenet_normalize, (224, 224))

    return model_tup, gradcam_resnet50_extractor
