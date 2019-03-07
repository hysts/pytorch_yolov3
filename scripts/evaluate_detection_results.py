#!/usr/bin/env python

import argparse
import json
import pathlib

from pycocotools.coco import COCO
from tensorboardX import SummaryWriter

import yolov3.utils.evaluation


def extract_image_ids(path):
    with open(path, 'r') as fin:
        gt_data = json.load(fin)
    image_ids = sorted([image['id'] for image in gt_data['images']])
    return image_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, required=True)
    parser.add_argument('--pred', type=str, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--outdir', type=str)
    args = parser.parse_args()

    coco = COCO(args.gt)
    image_ids = extract_image_ids(args.gt)
    stats = yolov3.utils.evaluation.evaluate(coco, args.pred, image_ids)

    if args.outdir is not None:
        outdir = pathlib.Path(args.outdir)
        outdir.mkdir(exist_ok=True, parents=True)
    else:
        outdir = pathlib.Path(args.pred).parent
    with open(outdir / 'stats.json', 'w') as fout:
        json.dump(list(stats), fout)

    writer = SummaryWriter(args.outdir)
    for name, val in zip(yolov3.utils.evaluation.stats_names, stats):
        writer.add_scalar(f'val/{name}', val, args.step)
    writer.close()


if __name__ == '__main__':
    main()
