import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F

from ia_utils.data_utils import imagenet_normalize


def pgd_attack_iter(model, x, y, cx, eps, alpha, num_steps, confidence):
    for i in range(num_steps):
        logits = model(imagenet_normalize(cx))
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

    return cx, flag


def pgd_attack(model, x, y, eps, alpha,
               x_adv_init=None, base_steps=500, inc_steps=100, max_steps=1000,
               confidence=None):
    x = torch.tensor(x)
    if x_adv_init is None:
        x_adv_init = x.clone()
    cx = torch.tensor(x_adv_init, requires_grad=True)

    cx, flag = pgd_attack_iter(model, x, y, cx, eps, alpha, base_steps, confidence)
    tot_steps = base_steps
    while not flag and tot_steps < max_steps:
        cx, flag = pgd_attack_iter(model, x, y, cx, eps, alpha,
                                   inc_steps, confidence)
        tot_steps += inc_steps

    return cx.detach(), flag, tot_steps
