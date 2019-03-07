#!/usr/bin/env python

import argparse
import json
import pathlib

import torch
import tqdm

import yolov3.models
import yolov3.utils.detection
from yolov3.config import get_default_config
from yolov3.utils.data.dataloader import create_dataloader
from yolov3.utils.data.postprocess import PostProcessor


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    if args.config is not None:
        config.merge_from_file(args.config)
    config.merge_from_list([
        'validation.outdir',
        args.outdir,
        'validation.ckpt_path',
        args.ckpt_path,
    ])
    config.merge_from_list(args.options)
    if not torch.cuda.is_available():
        config.validation.device = 'cpu'
    config.freeze()
    return config


def load_model(config):
    if config.model.name == 'yolov3':
        model = yolov3.models.YOLOv3(config)
    elif config.model.name == 'yolov3-tiny':
        model = yolov3.models.YOLOv3Tiny(config)
    else:
        raise ValueError(
            'config.model.name must be one of yolov3 or yolov3-tiny')
    ckpt = torch.load(config.validation.ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    return model


def main():
    config = load_config()

    device = torch.device(config.validation.device)

    model = load_model(config)
    model.to(device)
    model.eval()

    postprocessor = PostProcessor(
        conf_thresh=config.validation.conf_thresh,
        nms_thresh=config.validation.nms_thresh)

    dataloader = create_dataloader(config, is_train=False)

    detector = yolov3.utils.detection.Detector(
        model,
        postprocessor,
        config.validation.image_size,
        category_ids=dataloader.dataset.category_ids)

    detection_results = []
    for data, image_sizes, image_ids in tqdm.tqdm(dataloader):
        data = data.to(device)
        results = detector.detect(data, image_sizes, image_ids)
        detection_results += results

    outdir = pathlib.Path(config.validation.outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    outpath = outdir / 'predictions.json'
    with open(outpath, 'w') as fout:
        json.dump(detection_results, fout)


if __name__ == '__main__':
    main()
