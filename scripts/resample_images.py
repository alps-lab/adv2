#!/usr/bin/env python
import argparse
import csv
from shutil import copyfile
from pathlib import Path

import numpy as np
from progressbar import progressbar


def main(config):
    source_dir_path = Path(config.source_dir)
    with source_dir_path.joinpath("images.csv").open() as f:
        reader = csv.reader(f)
        rows = list(reader)
    rows = rows[1:]

    dest_dir_path = Path(config.dest_dir)
    dest_dir_path.mkdir(parents=True)

    rs = np.random.RandomState(config.seed)
    indices = rs.choice(np.arange(len(rows)), size=config.n, replace=False)
    subrows = [rows[index] for index in indices]

    with dest_dir_path.joinpath("images.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "label"])

        for row in progressbar(subrows):
            writer.writerow(row)
            copyfile(str(source_dir_path.joinpath(row[0])),
                     str(dest_dir_path.joinpath(row[0])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", metavar="SOURCE_DIR")
    parser.add_argument("dest_dir", metavar="DEST_DIR")
    parser.add_argument("n", metavar="N", type=int)
    parser.add_argument("-s", "--seed", type=int, dest="seed")

    main(parser.parse_args())
