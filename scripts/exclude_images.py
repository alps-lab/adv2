#!/usr/bin/env python
import argparse
import csv


def main(config):
    rows_1 = [tuple(row) for row in csv.reader(open(config.csv_1))]
    rows_2 = [tuple(row) for row in csv.reader(open(config.csv_2))]

    rows_new = rows_1[:1]
    rows_1 = rows_1[1:]
    rows_2 = set(rows_2[1:])

    for row in rows_1:
        if row not in rows_2:
            rows_new.append(row)

    with open(config.csv_save, 'w') as f:
        writer = csv.writer(f)
        for row in rows_new:
            writer.writerow(row)

    print('total excluded:', len(rows_1) - len(rows_new) + 1, 'total saved:', len(rows_new))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_1')
    parser.add_argument('csv_2')
    parser.add_argument('csv_save')
    main(parser.parse_args())
