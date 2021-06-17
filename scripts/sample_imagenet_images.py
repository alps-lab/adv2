#!/usr/bin/env python
from pathlib import Path
from shutil import copyfile
import argparse

import numpy as np


def main(config):
    all_paths = list(Path(config.source_dir).rglob("*.JPEG"))
    indices = np.random.RandomState().choice(len(all_paths), config.n, replace=False)
    paths = [all_paths[idx] for idx in indices]

    dest_dir_path = Path(config.dest_dir)
    dest_dir_path.mkdir(parents=True, exist_ok=True)
    for path in paths:
        dest_path = dest_dir_path.joinpath(path.name)
        copyfile(str(path), str(dest_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir")
    parser.add_argument("dest_dir")
    parser.add_argument("-n", dest="n",type=int, default=500)

    main(parser.parse_args())
