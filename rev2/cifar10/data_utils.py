import torch


CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]


def cifar10_normalize(x, mean=None, std=None):
    if mean is None:
        mean = CIFAR10_MEAN
    if std is None:
        std = CIFAR10_STD
    mu = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    st = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return (x - mu) / st
