#!/usr/bin/env python
import pickle

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics.classification import confusion_matrix, accuracy_score, f1_score, classification_report

from rev1.isic_acid.isic_model import get_isic_model_on_resnet50
from rev1.isic_acid.isic_utils import ISIC_EVAL_TRANSFORMS, ISIC_CV_SPLIT_PATH, ISICDataset, ISIC_RESNET50_CKPT_PATH


if __name__ == '__main__':
    indices = pickle.load(open(ISIC_CV_SPLIT_PATH, 'rb'))['valIndCV'][1]
    dataset = ISICDataset(indices=indices, transforms=ISIC_EVAL_TRANSFORMS)
    loader = DataLoader(dataset, batch_size=64, pin_memory=True)

    model = get_isic_model_on_resnet50(False, True, ISIC_RESNET50_CKPT_PATH)
    model.cuda()
    images = []
    preds = []
    truths = []
    paths = []
    for bx, by, bpath in loader:
        truths.append(by.numpy())
        images.append(bx.numpy())
        paths.extend(bpath)
        bx = bx.cuda(non_blocking=True)
        logits = model(bx)
        preds.append(logits.argmax(1).cpu().numpy())

    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    images = np.concatenate(images)
    mat = confusion_matrix(truths, preds)
    print('confusion matrix:\n', mat)
    print('accuracy score:', accuracy_score(truths, preds))
    print('f1 score:', f1_score(truths, preds, average='macro'))
    print('classification report:',
          classification_report(truths, preds))

    correct = preds == truths
    # images = images[correct]
    truths = truths[correct]
    paths = [paths[i] for i in np.nonzero(correct)[0]]
    print(truths[:5], paths[:5])

    df = pd.DataFrame.from_dict(dict(path=paths, label=truths))
    df.to_csv('/home/xinyang/Data/intattack/ISIC-2018/cv2_path.csv', index=False)
