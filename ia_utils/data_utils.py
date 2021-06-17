import json

import numpy as np
import cv2
import torch
import torchvision


IMAGENET_TRAIN_PATH = "/home/xinyang/Datasets/imagenet_1000"
IMAGENET_VAL_PATH = "/home/xinyang/Datasets/imagenet_val"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_LABELS = "/home/xinyang/Codes/intattack/resources/imagenet_class_index.json"

VGG19_NORMALIZE = torchvision.transforms.Normalize(
    mean=IMAGENET_MEAN, std=IMAGENET_STD)

TEST_TRANSFORM = torchvision.transforms.Compose(
    [torchvision.transforms.ToPILImage(),
     torchvision.transforms.Resize(256),
     torchvision.transforms.CenterCrop(224),
     torchvision.transforms.ToTensor(),
     VGG19_NORMALIZE
     ]
)

TEST_TRANSFORM_NO_NORMALIZE = torchvision.transforms.Compose(
    [torchvision.transforms.ToPILImage(),
     torchvision.transforms.Resize(256),
     torchvision.transforms.CenterCrop(224),
     torchvision.transforms.ToTensor(),
     ]
)


def imagenet_normalize(t, mean=None, std=None):
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std= IMAGENET_STD

    ts = []
    for i in range(3):
        ts.append(torch.unsqueeze((t[:, i] - mean[i]) / std[i], 1))
    return torch.cat(ts, dim=1)


def imagenet_denormalize(t):
    t = t.clone()
    t[:, 0] = t[:, 0] * 0.229 + 0.485
    t[:, 1] = t[:, 1] * 0.224 + 0.456
    t[:, 2] = t[:, 2] * 0.225 + 0.406

    return t


def transform_images(x, transform):
    arr = []
    for i in range(len(x)):
        arr.append(transform(x[i])[None])

    return torch.cat(arr, dim=0)


def load_imagenet_labels(path=None):
    path = path if path is not None else IMAGENET_LABELS
    with open(path) as f:
        dobj = json.load(f)

    ndobj = {}
    for key, value in dobj.items():
        ndobj[int(key)] = value
    return ndobj


def mask_to_image(mask, resize=None):
    """
    :param mask: torch.float/np.ndarray (h, w) [0, 1]
    :param resize: None or tuples (H, W)
    :return: np.ndarray (h, w) [0, 1]
    """
    if isinstance(mask, np.ndarray):
        m = mask.copy()
    else:
        m = mask.detach().cpu().numpy()

    mmin, mmax = m.min(), m.max()
    m_norm = (m - mmin) / mmax
    m = m_norm

    if resize is not None:
        m = cv2.resize(m, resize)

    return m


def cam_to_image(image, mask):
    """
    :param image: np.ndarray (c, h, w) [0, 1]
    :param mask: np.ndarray (h, w) [0, 1]
    :return: np.ndarray (c, h, w) [0, 1]
    """
    mask = np.uint8(255 * mask)
    mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    mask = mask.transpose([2, 0, 1])[::-1]
    blend = image + mask / 255.
    blend /= blend.max()

    return blend


def plot_mask(t, vis, resize=None, win=None, env=None):
    """
    :param t: torch.float/np.ndarray (h, w) [0, 1]
    """
    m = mask_to_image(t, resize)
    vis.image(np.uint8(m[None] * 255.), win=win, env=env)


def plot_cam(img, mask, vis, win=None, env=None):
    """
     :param img: np.ndarray (c, h, w) [0, 1]
     :param mask: np.ndarray (h, w) [0, 1]
    """
    blend = cam_to_image(img, mask)
    vis.image(np.uint8(blend * 255.), win=win, env=env, opts={"title": win})


def batch_iter(x, batch_size=64):
    i, l = 0, len(x)
    while i < l:
        s, t = i, i + batch_size
        t = min(t, l)
        yield (s, t), x[s:t]
        i += batch_size


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
