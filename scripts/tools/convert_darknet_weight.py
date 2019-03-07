#!/usr/bin/env python

import argparse
import pathlib
import numpy as np
import torch

import yolov3.models.darknet53
from yolov3.models import YOLOv3, YOLOv3Tiny
from yolov3.config import get_default_config


def load_bn(module, weight, offset):
    for name in ['bias', 'weight', 'running_mean', 'running_var']:
        param = getattr(module, name)
        size = param.numel()
        data = torch.from_numpy(weight[offset:offset + size]).view_as(param)
        param.data.copy_(data)
        offset += size
    return offset


def load_conv(module, weight, offset, bias):
    if bias:
        data = torch.from_numpy(
            weight[offset:offset + module.bias.numel()]).view_as(module.bias)
        module.bias.data.copy_(data)
        offset += module.bias.numel()

    data = torch.from_numpy(
        weight[offset:offset + module.weight.numel()]).view_as(module.weight)
    module.weight.data.copy_(data)
    offset += module.weight.numel()
    return offset


def load_convbn(module, weight, offset):
    offset = load_bn(module.bn, weight, offset)
    offset = load_conv(module.conv, weight, offset, bias=False)
    return offset


def load_darknet_bottleneck(module, weight, offset):
    offset = load_convbn(module.conv1, weight, offset)
    offset = load_convbn(module.conv2, weight, offset)
    return offset


def load_darknet_stage(module, weight, offset):
    offset = load_convbn(module.conv1, weight, offset)
    for index in range(1, len(module)):
        offset = load_darknet_bottleneck(module[index], weight, offset)
    return offset


def load_darknet(module, weight, offset):
    offset = load_convbn(module.conv1, weight, offset)
    offset = load_darknet_stage(module.stage1, weight, offset)
    offset = load_darknet_stage(module.stage2, weight, offset)
    offset = load_darknet_stage(module.stage3, weight, offset)
    offset = load_darknet_stage(module.stage4, weight, offset)
    offset = load_darknet_stage(module.stage5, weight, offset)
    return offset


def load_yolo_stage(module, weight, offset):
    offset = load_darknet_bottleneck(module.bottleneck1, weight, offset)
    offset = load_darknet_bottleneck(module.bottleneck2, weight, offset)
    offset = load_convbn(module.conv, weight, offset)
    return offset


def load_yolo_upsample(module, weight, offset):
    offset = load_convbn(module.conv, weight, offset)
    return offset


def load_yolo_layer(module, weight, offset):
    offset = load_conv(module.conv, weight, offset, bias=True)
    return offset


def load_yolo(module, weight, offset):
    offset = load_darknet(module.backbone, weight, offset)

    offset = load_yolo_stage(module.stage1, weight, offset)
    offset = load_convbn(module.conv1, weight, offset)
    offset = load_yolo_layer(module.yolo_layer1, weight, offset)

    offset = load_yolo_upsample(module.upsample1, weight, offset)
    offset = load_yolo_stage(module.stage2, weight, offset)
    offset = load_convbn(module.conv2, weight, offset)
    offset = load_yolo_layer(module.yolo_layer2, weight, offset)

    offset = load_yolo_upsample(module.upsample2, weight, offset)
    offset = load_yolo_stage(module.stage3, weight, offset)
    offset = load_convbn(module.conv3, weight, offset)
    offset = load_yolo_layer(module.yolo_layer3, weight, offset)
    return offset


def load_yolo_tiny(module, weight, offset):
    offset = load_convbn(module.conv1, weight, offset)
    offset = load_convbn(module.conv2, weight, offset)
    offset = load_convbn(module.conv3, weight, offset)
    offset = load_convbn(module.conv4, weight, offset)
    offset = load_convbn(module.conv5, weight, offset)
    offset = load_convbn(module.conv6, weight, offset)
    offset = load_convbn(module.conv7, weight, offset)
    offset = load_convbn(module.conv8, weight, offset)
    offset = load_convbn(module.conv9, weight, offset)
    offset = load_yolo_layer(module.yolo_layer1, weight, offset)
    offset = load_convbn(module.conv10, weight, offset)
    offset = load_convbn(module.conv11, weight, offset)
    offset = load_yolo_layer(module.yolo_layer2, weight, offset)
    return offset


def load_darknet_weight(path):
    with open(path, 'rb') as fin:
        _ = np.fromfile(fin, dtype=np.int32, count=5)
        weight = np.fromfile(fin, dtype=np.float32)
    return weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, required=True)
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        choices=['darknet53', 'yolov3', 'yolov3-tiny'])
    parser.add_argument('--rgb', dest='bgr', action='store_false')
    parser.add_argument('--outdir', type=str, required=True)
    args = parser.parse_args()

    config = get_default_config()
    if args.model_name in ['darknet53', 'yolov3']:
        config.merge_from_file('configs/yolov3.yaml')
    else:
        config.merge_from_file('configs/yolov3_tiny.yaml')
    config.freeze()

    weight = load_darknet_weight(args.weight_path)

    if args.model_name == 'darknet53':
        model = yolov3.models.darknet53.FeatureExtractor()
        offset = load_darknet(model, weight, 0)
        if args.bgr:
            param = list(model.conv1.conv.parameters())[0]
            param.data = param.data[:, [2, 1, 0]]
    elif args.model_name == 'yolov3':
        model = YOLOv3(config)
        offset = load_yolo(model, weight, 0)
        if args.bgr:
            param = list(model.backbone.conv1.conv.parameters())[0]
            param.data = param.data[:, [2, 1, 0]]
    elif args.model_name == 'yolov3-tiny':
        model = YOLOv3Tiny(config)
        offset = load_yolo_tiny(model, weight, 0)
        if args.bgr:
            param = list(model.conv1.conv.parameters())[0]
            param.data = param.data[:, [2, 1, 0]]
    else:
        raise ValueError()

    assert offset == len(weight), f'{offset} != {len(weight)}'

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    org_name = pathlib.Path(args.weight_path).name
    outname = f'{org_name}.pth'
    outpath = outdir / outname
    if args.model_name == 'darknet53':
        ckpt = {'model': model.state_dict()}
    else:
        ckpt = {'model': model.state_dict(), 'config': config}
    torch.save(ckpt, outpath)


if __name__ == '__main__':
    main()
