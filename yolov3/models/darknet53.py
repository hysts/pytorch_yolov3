import torch.nn as nn

from yolov3.models.common import ConvBN, DarknetStage, initialize


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBN(3, 32, kernel_size=3, stride=1, padding=1)
        self.stage1 = DarknetStage(32, 64, 1)
        self.stage2 = DarknetStage(64, 128, 2)
        self.stage3 = DarknetStage(128, 256, 8)
        self.stage4 = DarknetStage(256, 512, 8)
        self.stage5 = DarknetStage(512, 1024, 4)

        self.apply(initialize)

    def forward(self, x):
        outputs = []
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        outputs.append(x)
        x = self.stage4(x)
        outputs.append(x)
        x = self.stage5(x)
        outputs.append(x)
        return outputs
