#!/usr/bin/env python
import argparse
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import visdom
import cv2
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torchvision

from ia_utils.data_utils import imagenet_normalize, load_imagenet_labels
from ia.int_attacks.pgd import pgd_attack
from ia.int_models.gradcam import GradCam


def to_mask(m):
    m = m - m.min()
    m = m / m.max()
    return m


def make_cam(x, m):
    mask_x = cv2.applyColorMap(np.uint8(255. * cv2.resize(m, (224, 224))), cv2.COLORMAP_JET).transpose([2, 0, 1])[::-1]
    mask_x = np.float32(mask_x / 255.)
    cam_x = x + mask_x
    cam_x = cam_x / cam_x.max()
    return cam_x


def do_int_attack(gradcam_model, x, cx, target_label, mask_init,
                  eps, alpha, reg_ceoff, steps):
    best_cx, best_mask, best_int_loss_value = None, None, np.inf
    cx = torch.tensor(cx, requires_grad=True)

    mask_init_norm = mask_init - mask_init.min()
    mask_init_norm = mask_init_norm / mask_init_norm.max()

    for step in range(steps):
        y = torch.tensor([target_label]).long().cuda()
        logits, _, _, mask = gradcam_model(cx, y)
        current_label = np.asscalar(torch.max(logits, 1)[1])
        adv_loss = F.nll_loss(logits, y)

        mask_norm = mask - mask.min()
        mask_norm = mask_norm / mask_norm.max()
        int_loss = (mask_norm - mask_init_norm).abs().sum()

        adv_loss_value, int_loss_value = (np.asscalar(adv_loss.detach().cpu().numpy()),
                                          np.asscalar(int_loss.detach().cpu().numpy()))

        adv_grad = autograd.grad([adv_loss], [cx], retain_graph=True)[0]
        adv_grad_norm = torch.norm(adv_grad)

        if current_label == target_label and int_loss_value < best_int_loss_value:
            best_mask = mask.clone().detach()
            best_cx = cx.clone().detach()
            best_int_loss_value = int_loss_value

        int_grad = autograd.grad([int_loss], [cx], retain_graph=True)[0]
        int_grad_norm = torch.norm(int_grad)

        with torch.no_grad():
            cx = cx - alpha * adv_grad / adv_grad_norm - reg_ceoff * int_grad / int_grad_norm
            diff = cx - x
            diff.clamp_(-eps, eps)
            cx = x + diff
            cx.clamp_(0, 1)
        cx = torch.tensor(cx, requires_grad=True)

    return best_cx, best_mask, best_int_loss_value


def attack_one(gradcam_model, path, true_label, target_label=None, eps=0.02,
               debug=False, alpha=2. / 255, reg_coeff=0.04):
    model = gradcam_model.model

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

    if debug:
        vis.text("true label: %s, pred label: %s, target label: %s"
                 % (imagenet_labels[true_label][1], imagenet_labels[pred_label][1],
                    imagenet_labels[target_label][1]),
                 opts=dict(title="status_%s" % name),
                 win="status_%s" % name)

    x_adv, flag, num_steps = pgd_attack(model, x, target_label, eps=eps,
                                        alpha=alpha, base_steps=700,
                                        inc_steps=150, max_steps=1600,
                                        confidence=None)
    with torch.no_grad():
        logits = model(imagenet_normalize(x_adv))
    logits_np = logits.cpu().numpy()[0]
    res["logits_adv_x"] = logits_np
    x_adv_np = x_adv.cpu().numpy()[0]
    if debug:
        vis.image(np.uint8(x_adv_np * 255), win="adv_%s" % name,
                  opts=dict(title="adv_%s" % name))

    if not flag:
        res["status"] = -1  # -1: adv failed, 0: failed, 1: succeed
        return res

    res["status"] = 0
    res["pgd_adv"] = x_adv_np
    res["pgd_num_steps"] = num_steps

    prod_x = gradcam_model(x, true_label)[3].detach()
    mask_x = to_mask(prod_x.cpu().numpy()[0])
    cam_x = make_cam(cropped, mask_x)
    res["prod_x"] = prod_x.cpu().numpy()
    res["mask_x"] = mask_x[None]
    res["cam_x"] = cam_x
    if debug:
        vis.image(np.uint8(255. * cam_x), win="cam_x_%s" % name,
                  opts=dict(title="cam_x_%s" % name))

    prod_adv_x = gradcam_model(x_adv, target_label)[3].detach()
    mask_adv_x = to_mask(prod_adv_x.cpu().numpy()[0])
    cam_adv_x = make_cam(x_adv_np, mask_adv_x)
    res["prod_adv_x"] = prod_adv_x.cpu().numpy()
    res["mask_adv_x"] = mask_adv_x[None]
    res["cam_adv_x"] = cam_adv_x
    if debug:
        vis.image(np.uint8(255. * cam_adv_x), win="cam_adv_x_%s" % name,
                  opts=dict(title="cam_adv_x_%s" % name))

    best_cx, best_prod, best_int_loss = do_int_attack(gradcam_model, x, x_adv, target_label,
                                                      prod_x, eps, alpha, reg_coeff,
                                                      steps=1600)
    best_cx_np = best_cx.detach().cpu().numpy()[0]
    res["best_intadv_x"] = best_cx_np
    res["best_intadv_loss"] = best_int_loss
    if debug:
        vis.image(np.uint8(255. * best_cx_np), win="intadv_%s" % name,
                  opts=dict(title="intadv_%s" % name))
        vis.text("best_int_loss: %.6f" % best_int_loss, win="status_%s" % name,
                 append=True)
    best_prod = best_prod.detach()
    mask_intadv_x = to_mask(best_prod.cpu().numpy()[0])
    cam_intadv_x = make_cam(best_cx_np, mask_intadv_x)
    res["best_intadv_prod"] = best_prod.cpu().numpy()
    res["best_intadv_mask"] = mask_intadv_x[None]
    res["best_intadv_cam"] = cam_intadv_x
    if debug:
        vis.image(np.uint8(255. * cam_intadv_x), win="cam_intadv_x_%s" % name,
                  opts=dict(title="cam_intadv_x_%s" % name))

    with torch.no_grad():
        logits = model(imagenet_normalize(best_cx))
    logits_np = logits.cpu().numpy()[0]
    res["logits_best_intadv_x"] = logits_np
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

    model = torchvision.models.vgg19(pretrained=True)
    model.train(False)
    model.cuda()

    gradcam_model = GradCam(model,  target_layer_names=["35"])

    islice = slice(flags.start, None if flags.end == -1 else flags.end)
    num_attempts, num_succeed, num_run = 0, 0, 0
    for name, label in zip(names[islice], labels[islice]):
        start_time = datetime.now()
        if num_attempts > flags.n:
            print("stopped...")

        res = attack_one(gradcam_model, str(Path(flags.src_dir).joinpath(name)), label,
                         eps=flags.eps, debug=flags.debug)
        if res["status"] >= 0:
            num_attempts += 1
            if res["status"] == 1:
                num_succeed += 1
                print("succeed at: %s" % name)
            else:
                print("failed at: %s" % name)
        elif res["status"] == -1:
            print("fail to find PGD example at: %s" % name)
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
