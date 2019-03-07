#!/usr/bin/env python

import argparse
import json
import pathlib

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

import yolov3.models
import yolov3.utils.data.postprocess
import yolov3.utils.detection
from yolov3.config import get_default_config
from yolov3.transforms import create_transform


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ckpt', type=str, required=True, help='path to model to use')
    parser.add_argument(
        '--config', type=str, help='path to configuration file')
    parser.add_argument(
        '--image', type=str, default='', help='path to an image to test')
    parser.add_argument(
        '--image_dir', type=str, default='', help='path to an image directory')
    parser.add_argument(
        '--video', type=str, default='', help='path to a video to test')
    parser.add_argument(
        '--camera',
        action='store_true',
        help='with this option, demo will be run on camera input')
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    if args.config is not None:
        config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    config.merge_from_list([
        'demo.image_path',
        args.image,
        'demo.image_dir',
        args.image_dir,
        'demo.video_path',
        args.video,
        'demo.camera',
        args.camera,
        'demo.ckpt_path',
        args.ckpt,
    ])
    if not torch.cuda.is_available():
        config.demo.device = 'cpu'
    config.freeze()
    return config


def load_model(config):
    ckpt = torch.load(config.demo.ckpt_path, map_location='cpu')
    if config.model.name != ckpt['config'].model.name:
        ckpt_config = ckpt['config']
        raise RuntimeError(
            'config.model.name is different from that of checkpoint config: '
            f'{config.model.name} != {ckpt_config.model.name}\n'
            f'You need to specify correct configuration file')

    if config.model.name == 'yolov3':
        model = yolov3.models.YOLOv3(config)
    elif config.model.name == 'yolov3-tiny':
        model = yolov3.models.YOLOv3Tiny(config)
    else:
        raise ValueError(
            'config.model.name must be one of yolov3 or yolov3-tiny')
    model.load_state_dict(ckpt['model'])
    return model


def load_class_names(config):
    with open(config.demo.category_names, 'r') as fin:
        names = json.load(fin)
    return names


def generate_colors(num_classes):
    np.random.seed(0)
    return np.random.randint(128, 256, size=(num_classes, 3))


def visualize_bboxes(image,
                     bboxes,
                     labels,
                     scores,
                     colors,
                     alpha=1.,
                     linewidth=3.,
                     ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    ax.imshow(image)

    if len(bboxes) == 0:
        return ax

    for bbox, color, label, score in zip(bboxes, colors, labels, scores):
        tl = tuple(bbox[:2])
        width, height = bbox[2:4]
        ax.add_patch(
            plt.Rectangle(
                tl,
                width,
                height,
                fill=False,
                edgecolor=color / 255,
                linewidth=linewidth,
                alpha=alpha))

        caption = f'{label}: {score:.02f}'

        ax.text(
            tl[0],
            tl[1],
            caption,
            style='italic',
            bbox={
                'facecolor': 'white',
                'alpha': 0.7,
                'pad': 10
            })
    return ax


class Detector(yolov3.utils.detection.Detector):
    def __init__(self, config):
        self.config = config
        self.class_names = load_class_names(config)
        self.class_colors = generate_colors(len(self.class_names))
        self.device = torch.device(config.demo.device)
        self.transform = create_transform(config, is_train=False)

        model = load_model(config)
        model.to(self.device)
        model.eval()

        postprocessor = yolov3.utils.data.postprocess.PostProcessor(
            config.demo.conf_thresh, config.demo.nms_thresh)

        super().__init__(model, postprocessor, config.demo.image_size)

    def detect_and_draw(self, image, image_id=0):
        if self.config.demo.channel_order != 'bgr':
            image = image[:, :, ::-1]
        h, w = image.shape[:2]
        image_size = torch.tensor([w, h]).float()

        data, _ = self.transform(image, None)
        data = data.unsqueeze(0).to(self.device)

        detections = super().detect(data, [image_size],
                                    torch.tensor([image_id]))

        bboxes = []
        scores = []
        names = []
        colors = []
        for detection in detections:
            bboxes.append(detection['bbox'])
            scores.append(detection['score'])
            class_index = detection['category_index']

            name = self.class_names[class_index]
            names.append(name)
            colors.append(self.class_colors[class_index])
            print(f'{name:<12s} score: {scores[-1]:.5f}, bbox: {bboxes[-1]}')
        names = np.array(names)
        colors = np.array(colors)

        if self.config.demo.channel_order == 'bgr':
            image = image[:, :, ::-1]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        visualize_bboxes(
            image,
            bboxes=bboxes,
            labels=names,
            scores=scores,
            colors=colors,
            linewidth=2,
            ax=ax)
        plt.axis('off')
        fig.tight_layout()
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plt.close()
        res = data.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
        return res


def main():
    config = load_config()
    print(str(config))

    detector = Detector(config)

    if config.demo.camera or config.demo.video_path != '':
        if config.demo.camera:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(config.demo.video_path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        count = 0
        while True:
            key = cv2.waitKey(1) & 0xff
            if key in [27 or ord('q')]:
                break
            ok, frame = cap.read()
            if not ok:
                break
            if config.demo.camera:
                frame = frame[:, ::-1]
            if config.demo.camera:
                print(f'frame {count + 1}')
            else:
                print(f'frame {count + 1}/{n_frames}')
            res = detector.detect_and_draw(frame)
            cv2.imshow('results', res[:, :, ::-1])
            count += 1
        cap.release()
    elif config.demo.image_dir != '':
        image_dir = pathlib.Path(config.demo.image_dir)
        paths = [
            path for path in sorted(image_dir.glob('*'))
            if path.name.split('.')[-1].lower() in ['jpg', 'png']
        ]
        for path in paths:
            image = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
            print(path)
            res = detector.detect_and_draw(image)
            cv2.imshow('results', res[:, :, ::-1])
            key = cv2.waitKey() & 0xff
            if key in [27, ord('q')]:
                break
    else:
        image = cv2.imread(config.demo.image_path, cv2.IMREAD_COLOR)
        print(config.demo.image_path)
        res = detector.detect_and_draw(image)
        cv2.imshow('results', res[:, :, ::-1])
        cv2.waitKey()


if __name__ == '__main__':
    main()
