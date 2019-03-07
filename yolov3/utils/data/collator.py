import cv2
import numpy as np
import torch

from yolov3.transforms import Compose, Normalize, ToTensor


class Collator:
    def __init__(self, input_size, min_size, max_size):
        self.input_size = input_size
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, batch):
        out_size = _generate_size(self.min_size, self.max_size)
        transform = Compose([
            # Here, the targets are supposed to be normalized by the
            # image size, so we change only the images
            ResizeImage(out_size=out_size),
            Normalize(),
            ToTensor(),
        ])
        batch = [transform(*data) for data in batch]
        batch = tuple([torch.stack(data, dim=0) for data in list(zip(*batch))])
        return batch


def _generate_size(min_size, max_size):
    min_index = min_size // 32
    max_index = max_size // 32
    return 32 * np.random.randint(min_index, max_index + 1)


class ResizeImage:
    def __init__(self, out_size):
        self.out_size = out_size

    def __call__(self, image, targets):
        image = cv2.resize(image, (self.out_size, self.out_size))
        return image, targets
