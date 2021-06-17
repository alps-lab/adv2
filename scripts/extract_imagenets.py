#!/usr/bin/env python
import argparse
import os
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir")
    parser.add_argument("target_dir")

    flags = parser.parse_args()
    assert flags.source_dir != flags.target_dir

    with open("imagenet_class_index.json") as f:
        dobj = json.load(f)

    os.makedirs(flags.target_dir, exist_ok=True)

    source_paths, target_paths = [], []
    for value in dobj.values():
        name, _ = value
        source_path = os.path.join(flags.source_dir, "%s.tar" % name)
        target_path = os.path.join(flags.target_dir, "%s.tar" % name)
        assert os.path.exists(source_path)
        source_paths.append(source_path)
        target_paths.append(target_path)

    for source_path, target_path in zip(source_paths, target_paths):
        os.link(source_path, target_path)
