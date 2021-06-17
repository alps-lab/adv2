#!/usr/bin/env python
import argparse
import csv


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source_path')
    parser.add_argument('dest1_path')
    parser.add_argument('dest2_path')
    parser.add_argument('cut', type=int)

    config = parser.parse_args()
    with open(config.source_path) as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
    header = rows[0]
    rows = rows[1:]

    rows_d1 = rows[:config.cut]
    rows_d2 = rows[config.cut:]
    with open(config.dest1_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows_d1)
    with open(config.dest2_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows_d2)
