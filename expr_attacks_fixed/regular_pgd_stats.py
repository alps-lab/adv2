#!/usr/bin/env python
import argparse

import numpy as np
import torch
import torch.nn.functional as F


def main(data_paths):
    keys = ['pgd_step_%d_adv_x', 'pgd_step_%d_logits_adv', 'pgd_step_%d_succeed']
    target_step = 1200
    keys = [key % target_step for key in keys]

    dobj = {key: [] for key in keys}
    for path in data_paths:
        h = np.load(path)
        for key in keys:
            dobj[key].append(h[key])
    dobj = {key: np.concatenate(value, axis=0) for key, value in dobj.items()}

    succeed = dobj[keys[-1]]
    logits = dobj[keys[1]]
    probs = F.softmax(torch.from_numpy(logits), dim=-1).max(1)[0].numpy()

    print('succeed', np.mean(succeed))
    print('confidence', np.mean(probs[succeed == 1]), np.std(probs[succeed == 1], ddof=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_paths', nargs='+')
    main(parser.parse_args().data_paths)