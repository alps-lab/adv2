import os
import csv

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, RandomCrop


ISIC_EVAL_TRANSFORMS = Compose([
                                CenterCrop(224),
                                ToTensor()]
                               )
ISIC_IMAGES_DIR = '/home/xinyang/Data/intattack/ISIC-2018/data/isic/task3/images/HAM10000/'
ISIC_LABEL_PATH = '/home/xinyang/Data/intattack/ISIC-2018/data/isic/task3/labels/HAM10000/labels.csv'
ISIC_CV_SPLIT_PATH = '/home/xinyang/Data/intattack/ISIC-2018/data/isic/indices_new.pkl'

ISIC_RESNET50_CKPT_PATH = '/home/xinyang/Data/intattack/ISIC-2018/data/isic/ISIC.example_resnet50_5foldcv/CVSet1/checkpoint_best-65.pt'


def read_metadata(path):
    reader = csv.reader(open(path))
    it = iter(reader)
    next(it)
    images, labels = [], []
    for row in it:
        image, label = row[0], [int(float(v)) for v in row[1:]]
        label = label.index(1)
        images.append(image + '.jpg')
        labels.append(label)
    return images, np.asarray(labels, np.int64)


class ISICDataset(Dataset):

    def __init__(self, images_dir=None, label_path=None, indices=None, transforms=None):
        if images_dir is None:
            images_dir = ISIC_IMAGES_DIR
        self.image_dir = images_dir
        if label_path is None:
            label_path = ISIC_LABEL_PATH
        self.image_paths, self.labels = read_metadata(label_path)
        if indices is not None:
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels = self.labels[indices]
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = os.path.join(self.image_dir, self.image_paths[index])
        image = Image.open(path)
        if self.transforms is not None:
            image = self.transforms(image)
        label = self.labels[index]
        return image, label, path
