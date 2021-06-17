#!/usr/bin/env python
from pathlib import Path
import csv
from datetime import datetime
import argparse

import visdom
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from ia_utils.data_utils import imagenet_normalize, load_imagenet_labels
from ia_utils.model_utils import resnet152
from ia.int_models.cam import CAM


def to_mask(m):
    m = m - m.min()
    m /= m.max()
    return m


def make_cam(x, m):
    mask_x = cv2.applyColorMap(np.uint8(255. * cv2.resize(m, (224, 224))), cv2.COLORMAP_JET).transpose([2, 0, 1])[::-1]
    mask_x = np.float32(mask_x / 255.)
    cam_x = x + mask_x
    cam_x = cam_x / cam_x.max()
    return cam_x


def pgd_attack_iter(cam_model, x, y, cx, eps, alpha, num_steps, confidence):
    model = cam_model.model
    cx_aux, mask_aux, logits_aux = [], [], []
    for i in range(num_steps):
        if i % 10 == 9:
            cx_aux.append(cx.detach().cpu().numpy())
            prod_x = cam_model(imagenet_normalize(cx), y)[1]
            mask_x = to_mask(prod_x.detach().cpu().numpy()[0])
            mask_aux.append(mask_x[None])

        logits = model(imagenet_normalize(cx))
        if i % 10 == 9:
            logits_aux.append(logits.detach().cpu().numpy())

        loss = F.nll_loss(logits, torch.tensor([y]).cuda())
        grad = autograd.grad([loss], [cx])[0]
        with torch.no_grad():
            grad_norm = torch.norm(grad)
            cx = cx - alpha * grad / grad_norm
            diff = cx - x
            diff.clamp_(-eps, eps)
            cx = x + diff
            cx.clamp_(0, 1)

        cx = torch.tensor(cx, requires_grad=True)

    with torch.no_grad():
        logits = model(imagenet_normalize(cx))
        logits_np = logits.cpu().numpy()[0]
        logits_arg = np.argsort(logits_np).astype(np.int)
        label = logits_arg[-1]

    if confidence is None:
        flag = label == y
    else:
        flag = (logits_np[label] - logits_np[logits_arg[-2]]) > confidence

    cx_aux = np.concatenate(cx_aux, axis=0)
    mask_aux = np.concatenate(mask_aux, axis=0)
    logits_aux = np.concatenate(logits_aux, axis=0)
    return cx, flag, cx_aux, mask_aux, logits_aux


def pgd_attack(cam_model, x, y, eps, alpha,
               x_adv_init=None, base_steps=500, inc_steps=100, max_steps=1000,
               confidence=None):
    model = cam_model.model
    x = torch.tensor(x)
    if x_adv_init is None:
        x_adv_init = x.clone()
    cx = torch.tensor(x_adv_init, requires_grad=True)

    cx, flag, cx_auxs, mask_auxs, logits_auxs = pgd_attack_iter(cam_model, x, y, cx, eps, alpha, base_steps, confidence)
    tot_steps = base_steps
    while not flag and tot_steps < max_steps:
        cx, flag, cx_aux, mask_aux, logits_aux = pgd_attack_iter(cam_model, x, y, cx, eps, alpha,
                                                                 inc_steps, confidence)
        tot_steps += inc_steps
        cx_auxs = np.concatenate([cx_auxs, cx_aux], axis=0)
        mask_auxs = np.concatenate([mask_auxs, mask_aux], axis=0)
        logits_auxs = np.concatenate([logits_auxs, logits_aux], axis=0)

    return cx.detach(), flag, tot_steps, cx_auxs, mask_auxs, logits_auxs


def attack_one(cam_model, path, true_label, target_label=None, eps=0.02, debug=False,
               alpha=2. / 255):
    model = cam_model.model

    name = Path(path).name.split(".")[0]
    res = {}
    img = cv2.imread(path)
    cropped = cv2.resize(img, (224, 224))

    cropped = cropped.transpose([2, 0, 1])[::-1]
    if debug:
        vis.image(cropped, win="cropped_%s" % name,
                  opts=dict(title="cropped_%s" % name))
    cropped = np.float32(cropped / 255.)
    res["cropped"] = cropped
    res["true_label"] = true_label

    x = torch.tensor(cropped[None]).cuda()
    x = torch.tensor(x, requires_grad=True)
    with torch.no_grad():
        logits = model(imagenet_normalize(x))
    logits_np = logits.cpu().numpy()[0]
    logits_arg = np.argsort(logits_np).astype(np.int)
    pred_label = int(logits_arg[-1])
    res["pred_label"] = pred_label
    res["logits_x"] = logits_np
    if pred_label != true_label:
        res["status"] = -2
        return res

    if target_label is None:
        target_label = int(logits_arg[0])
    res["target_label"] = target_label

    prod_x = cam_model(imagenet_normalize(x), true_label)[1]
    mask_x = to_mask(prod_x.detach().cpu().numpy()[0])
    res["prod_x"] = prod_x.detach().cpu().numpy()[0]
    res["mask_x"] = mask_x[None]

    if debug:
        vis.text("true label: %s, pred label: %s, target label: %s"
                 % (imagenet_labels[true_label][1], imagenet_labels[pred_label][1],
                    imagenet_labels[target_label][1]),
                 opts=dict(title="status_%s" % name),
                 win="status_%s" % name)

    x_adv, flag, num_steps, cx_auxs, mask_auxs, logits_auxs = pgd_attack(cam_model, x, target_label, eps=eps,
                                                                         alpha=alpha, base_steps=700,
                                                                         inc_steps=150, max_steps=1600,
                                                                         confidence=None)
    with torch.no_grad():
        logits = model(imagenet_normalize(x_adv))
    logits_np = logits.cpu().numpy()[0]
    res["logits_adv_x"] = logits_np
    res["x_aux"] = cx_auxs
    res["mask_aux"] = mask_auxs
    res["logits_aux"] = logits_auxs

    prod_adv_x = cam_model(imagenet_normalize(x_adv), target_label)[1]
    mask_adv_x = to_mask(prod_adv_x.detach().cpu().numpy()[0])
    res["prod_adv_x"] = prod_adv_x.detach().cpu().numpy()[0]
    res["mask_adv_x"] = mask_adv_x[None]

    x_adv_np = x_adv.cpu().numpy()[0]
    if debug:
        vis.image(np.uint8(x_adv_np * 255), win="adv_%s" % name,
                  opts=dict(title="adv_%s" % name))

    if not flag:
        res["status"] = 0  # 0: adv failed, 1: adv succeed
        return res

    res["status"] = 1
    return res


def main(flags):
    with Path(flags.src_dir).joinpath("images.csv").open() as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
    dest_dir_path = Path(flags.dest_dir)
    dest_dir_path.mkdir(exist_ok=True)

    names, labels = [], []
    for name, label in rows[1:]:
        names.append(name)
        labels.append(int(label))
    labels = np.asarray(labels, np.int)

    model = resnet152(pretrained=True)
    model.train(False)
    model.cuda()

    cam_model = CAM(model)

    islice = slice(flags.start, None if flags.end == -1 else flags.end)
    num_attempts, num_succeed, num_run = 0, 0, 0
    for name, label in zip(names[islice], labels[islice]):
        start_time = datetime.now()
        if num_attempts >= flags.n:
            print("stopped...")
            break

        res = attack_one(cam_model, str(Path(flags.src_dir).joinpath(name)), label,
                   eps=flags.eps, debug=False)
        if res["status"] >= 0:
            num_attempts += 1
            if res["status"] == 1:
                num_succeed += 1
                print("succeed at: %s" % name)
            else:
                print("failed at: %s" % name)
        elif res["status"] == -2:
            print("fail to predict at: %s" % name)

        np.savez(str(dest_dir_path.joinpath(name.split(".")[0])) + ".npz",
                 **res)
        end_time = datetime.now()
        num_run += 1
        print("time passed: %d, total run: %d" % ((end_time - start_time).seconds, num_run))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir", metavar="SRC_DIR")
    parser.add_argument("dest_dir", metavar="DEST_DIR")
    parser.add_argument("-s", "--start", default=0, type=int, dest="start")
    parser.add_argument("-e", "--end", default=-1, type=int, dest="end")
    parser.add_argument("-n", default=1000, type=int, dest="n")
    parser.add_argument("--eps", dest="eps", type=float, default=0.02)
    parser.add_argument("-v", "--verbose", action="store_true", dest="debug")
    parser.add_argument("--env", default="gradcam1000", dest="env")

    flags = parser.parse_args()

    vis = visdom.Visdom(env=flags.env)
    imagenet_labels = load_imagenet_labels()

    main(flags)
