#!/usr/bin/env python
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

from expr_detect.lid_utilities import get_big_batch_features, extract_lid_features


def test_lid_model(model_tup, lr_model_tup, train_x, test_x, test_y, k, cuda):
    n_features = 20
    train_x = train_x[np.random.choice(len(train_x), min(len(train_x), 100), replace=False)]
    features_train_x = get_big_batch_features(model_tup, train_x, n_features, cuda)
    features_test_x = get_big_batch_features(model_tup, test_x, n_features, cuda)

    mle_func = lambda v: -k / np.sum(np.log(v / v[-1]))
    dist_mats = [f[:, 1:k + 1] for f in extract_lid_features(features_train_x, features_test_x)]
    lid_mle = np.concatenate([np.apply_along_axis(mle_func, 1, f)[:, None] for f in dist_mats], axis=1)
    lr_test_x = lid_mle
    lr_test_y = test_y
    scalar, lr_model = lr_model_tup
    predict_proba = lr_model.predict_proba(scalar.transform(lr_test_x))
    preds = np.argmax(predict_proba, axis=1)
    auc_score = roc_auc_score(lr_test_y, predict_proba[:, lr_model.classes_.tolist().index(1)])
    confusion_mat = confusion_matrix(lr_test_y, preds)

    return auc_score, confusion_mat
