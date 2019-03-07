import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F

from yolov3.models.common import ConvBN, DarknetBottleneck, initialize
from yolov3.models.yolo_layer import YOLOLayer


class YOLOv3Stage(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.add_module(
            'bottleneck1',
            DarknetBottleneck(in_channels, 2 * out_channels, shortcut=False))
        self.add_module(
            'bottleneck2',
            DarknetBottleneck(
                2 * out_channels, 2 * out_channels, shortcut=False))
        self.add_module(
            'conv',
            ConvBN(
                2 * out_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0))


class YOLOv3Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = ConvBN(
            in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = F.interpolate(self.conv(x), scale_factor=2, mode='nearest')
        return x


class YOLOv3Base(nn.Module):
    def __init__(self, backbone_name):
        super().__init__()
        module = importlib.import_module(f'yolov3.models.{backbone_name}')
        self.backbone = getattr(module, 'FeatureExtractor')()

        outputs = self.backbone(
            torch.zeros((1, 3, 64, 64), dtype=torch.float32))
        skip_channels = [output.size(1) for output in outputs]
        in_channels1 = skip_channels[2]
        out_channels1 = in_channels1 // 2
        in_channels2 = skip_channels[1] + out_channels1 // 2
        out_channels2 = out_channels1 // 2
        in_channels3 = skip_channels[0] + out_channels2 // 2
        out_channels3 = out_channels2 // 2

        self.stage1 = YOLOv3Stage(in_channels1, out_channels1)
        self.upsample1 = YOLOv3Upsample(out_channels1)
        self.stage2 = YOLOv3Stage(in_channels2, out_channels2)
        self.upsample2 = YOLOv3Upsample(out_channels2)
        self.stage3 = YOLOv3Stage(in_channels3, out_channels3)

    def forward(self, x):
        feature_stage3, feature_stage4, feature_stage5 = self.backbone(x)
        outputs = []
        x = self.stage1(feature_stage5)
        outputs.append(x)
        x = torch.cat([self.upsample1(x), feature_stage4], dim=1)
        x = self.stage2(x)
        outputs.append(x)
        x = torch.cat([self.upsample2(x), feature_stage3], dim=1)
        x = self.stage3(x)
        outputs.append(x)
        return outputs


class YOLOv3(YOLOv3Base):
    def __init__(self, config):
        super().__init__(config.model.backbone)
        outputs = super().forward(
            torch.zeros((1, 3, 64, 64), dtype=torch.float32))
        n_channels = [output.size(1) for output in outputs]
        self.conv1 = ConvBN(
            n_channels[0],
            n_channels[0] * 2,
            kernel_size=3,
            stride=1,
            padding=1)
        self.conv2 = ConvBN(
            n_channels[1],
            n_channels[1] * 2,
            kernel_size=3,
            stride=1,
            padding=1)
        self.conv3 = ConvBN(
            n_channels[2],
            n_channels[2] * 2,
            kernel_size=3,
            stride=1,
            padding=1)
        all_anchors = torch.tensor(config.model.anchors)
        all_anchor_indices = torch.tensor(config.model.anchor_indices)

        self.yolo_layer1 = YOLOLayer(
            n_channels[0] * 2,
            all_anchors=all_anchors,
            anchor_indices=all_anchor_indices[2],
            downsample_rate=32,
            n_classes=config.data.n_classes,
            iou_thresh=config.train.iou_thresh)
        self.yolo_layer2 = YOLOLayer(
            n_channels[1] * 2,
            all_anchors=all_anchors,
            anchor_indices=all_anchor_indices[1],
            downsample_rate=16,
            n_classes=config.data.n_classes,
            iou_thresh=config.train.iou_thresh)
        self.yolo_layer3 = YOLOLayer(
            n_channels[2] * 2,
            all_anchors=all_anchors,
            anchor_indices=all_anchor_indices[0],
            downsample_rate=8,
            n_classes=config.data.n_classes,
            iou_thresh=config.train.iou_thresh)

        self.apply(initialize)

    def forward(self, x, targets=None):
        outputs = super().forward(x)
        out1 = self.yolo_layer1(self.conv1(outputs[0]), targets)
        out2 = self.yolo_layer2(self.conv2(outputs[1]), targets)
        out3 = self.yolo_layer3(self.conv3(outputs[2]), targets)
        if targets is None:
            return torch.cat([out1, out2, out3], dim=1)
        else:
            return torch.stack([*out1, *out2, *out3]).view(3, 5).sum(dim=0)
