import torch
import torch.nn as nn
import torch.nn.functional as F

from yolov3.models.common import ConvBN, initialize
from yolov3.models.yolo_layer import YOLOLayer


class YOLOv3Tiny(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.conv1 = ConvBN(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBN(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = ConvBN(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = ConvBN(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = ConvBN(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = ConvBN(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv7 = ConvBN(512, 1024, kernel_size=3, stride=1, padding=1)

        all_anchors = torch.tensor(config.model.anchors)
        all_anchor_indices = torch.tensor(config.model.anchor_indices)

        self.conv8 = ConvBN(1024, 256, kernel_size=1, stride=1, padding=0)
        self.conv9 = ConvBN(256, 512, kernel_size=3, stride=1, padding=1)
        self.yolo_layer1 = YOLOLayer(
            512,
            all_anchors=all_anchors,
            anchor_indices=all_anchor_indices[1],
            downsample_rate=32,
            n_classes=config.data.n_classes,
            iou_thresh=config.train.iou_thresh)

        self.conv10 = ConvBN(256, 128, kernel_size=1, stride=1, padding=0)
        self.conv11 = ConvBN(384, 256, kernel_size=3, stride=1, padding=1)
        self.yolo_layer2 = YOLOLayer(
            256,
            all_anchors=all_anchors,
            anchor_indices=all_anchor_indices[0],
            downsample_rate=16,
            n_classes=config.data.n_classes,
            iou_thresh=config.train.iou_thresh)

        self.apply(initialize)

    def forward(self, x, targets=None):
        x = F.max_pool2d(self.conv1(x), kernel_size=2, stride=2)
        x = F.max_pool2d(self.conv2(x), kernel_size=2, stride=2)
        x = F.max_pool2d(self.conv3(x), kernel_size=2, stride=2)
        x = F.max_pool2d(self.conv4(x), kernel_size=2, stride=2)
        y = self.conv5(x)
        x = F.max_pool2d(y, kernel_size=2, stride=2)
        x = F.max_pool2d(
            F.pad(self.conv6(x), pad=[0, 1, 0, 1]), kernel_size=2, stride=1)

        x = self.conv7(x)
        z = self.conv8(x)

        x = self.conv9(z)
        out1 = self.yolo_layer1(x, targets)

        x = F.interpolate(self.conv10(z), scale_factor=2, mode='nearest')
        x = torch.cat([x, y], dim=1)
        x = self.conv11(x)
        out2 = self.yolo_layer2(x, targets)
        if targets is None:
            return torch.cat([out1, out2], dim=1)
        else:
            return torch.stack([*out1, *out2]).view(2, 5).sum(dim=0)
