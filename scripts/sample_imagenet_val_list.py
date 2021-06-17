#!/usr/bin/env python
import csv
import argparse
from collections import Counter
from pathlib import Path
from shutil import copyfile

import numpy as np
from progressbar import progressbar
from ia_utils.data_utils import load_imagenet_labels


def main(flags):
    rs = np.random.RandomState(flags.seed)
    src_paths = sorted(Path(flags.src_dir).rglob("*.JPEG"))
    counter = Counter()
    for src_path in src_paths:
        cid = src_path.parent.name
        counter[cid] += 1

    probs = []
    for src_path in src_paths:
        cid = src_path.parent.name
        probs.append(1. / counter[cid])
    prob = np.asarray(probs)
    prob = prob / prob.sum()
    picked_indices = rs.choice(np.arange(len(src_paths)), flags.n, replace=False, p=prob)
    picked_paths = [src_paths[index] for index in picked_indices]

    dest_dir_path = Path(flags.dest_dir)
    dest_dir_path.mkdir(exist_ok=True)

    imagenet_labels = load_imagenet_labels()
    imagenet_labels = {v[0]: k for k, v in imagenet_labels.items()}

    csv_path = Path(dest_dir_path).joinpath("images.csv")
    with csv_path.open("w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["name", "label"])
        for picked_path in progressbar(picked_paths):
            cid = picked_path.parent.name
            label = str(imagenet_labels[cid])
            target_name = "%s_%s" % (cid, picked_path.name)
            writer.writerow([target_name, label])
            copyfile(str(picked_path), str(dest_dir_path.joinpath(target_name)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir", metavar="SOURCE_DIR")
    parser.add_argument("dest_dir", metavar="DEST_DIR")
    parser.add_argument("n", type=int, metavar="N")
    parser.add_argument("-s", "--seed", type=int, dest="seed")
    main(parser.parse_args())
