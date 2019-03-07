import random
import cv2
import numpy as np
import torch


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, targets):
        for transform in self.transforms:
            image, targets = transform(image, targets)
        return image, targets


class FlipColorChannelOrder:
    def __call__(self, image, targets=None):
        image = image[:, :, ::-1]
        return image, targets


class Normalize:
    def __call__(self, image, targets=None):
        image = image.astype(np.float32) / 255
        return image, targets


class NormalizeTargets:
    def __call__(self, image, targets=None):
        size = max(image.shape[:2])
        if targets is not None:
            targets[:, :4] /= size
        return image, targets


class Pad:
    def __init__(self, random_padding):
        self.random_padding = random_padding

    def __call__(self, image, targets=None):
        h, w = image.shape[:2]
        offset = self._compute_offset(np.array([w, h]))
        left, top = offset
        out_size = max(w, h)
        new_image = np.ones((out_size, out_size, 3), dtype=np.uint8) * 127
        new_image[top:top + h, left:left + w] = image
        if targets is not None:
            targets[:, :2] += offset
        return new_image, targets

    def _compute_offset(self, image_size):
        out_size = max(image_size)
        w, h = image_size
        if self.random_padding:
            dx = np.random.randint(out_size - w + 1)
            dy = np.random.randint(out_size - h + 1)
        else:
            dx = (out_size - w) // 2
            dy = (out_size - h) // 2
        return np.array([dx, dy])


class RandomDistort:
    def __init__(self, hue, saturation, exposure):
        self.hue = hue
        self.saturation = saturation
        self.exposure = exposure

    def __call__(self, image, targets=None):
        image = self._randomly_distort(image)
        return image, targets

    @staticmethod
    def _compute_random_scale(max_scale):
        scale = np.random.uniform(1, max_scale)
        return scale if random.random() < 0.5 else 1 / scale

    def _randomly_distort(self, image):
        diff_hue = np.random.uniform(-self.hue, self.hue)
        sat_scale = self._compute_random_scale(self.saturation)
        exp_scale = self._compute_random_scale(self.exposure)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 0] /= 179
        hsv[:, :, 1:] /= 255
        hue = hsv[:, :, 0] + diff_hue
        hsv[:, :, 1] *= sat_scale
        hsv[:, :, 2] *= exp_scale
        if diff_hue > 0:
            hue[hue > 1] -= 1
        else:
            hue[hue < 0] += 1
        hsv[:, :, 0] = np.round((hue * 179).clip(0, 179))
        hsv[:, :, 1:] = np.round((hsv[:, :, 1:] * 255).clip(0, 255))
        hsv = hsv.astype(np.uint8)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return image


class RandomHorizontalFlip:
    def __call__(self, image, targets=None):
        if random.random() < 0.5:
            image = image[:, ::-1]
            if targets is not None:
                targets[:, 0] = 1 - targets[:, 0]
        return image, targets


class Resize:
    def __init__(self, out_size, random_aspect_ratio_jitter):
        self.out_size = out_size
        self.jitter = random_aspect_ratio_jitter

    def __call__(self, image, targets=None):
        h, w = image.shape[:2]
        org_size = np.array([w, h])
        new_size = self.jitter_and_scale_size(org_size)
        image = cv2.resize(image, tuple(new_size))
        if targets is not None:
            scale = new_size / org_size
            targets[:, :4] *= np.concatenate([scale, scale])
        return image, targets

    def jitter_and_scale_size(self, image_size):
        w, h = self._jitter_size(image_size, self.jitter)
        new_size = self._scale_size_based_on_longer_edge(self.out_size, w / h)
        return new_size

    @staticmethod
    def _jitter_size(image_size, jitter):
        max_dw, max_dh = image_size * jitter
        dw = np.random.uniform(-max_dw, max_dw)
        dh = np.random.uniform(-max_dh, max_dh)
        return image_size + np.array([dw, dh])

    @staticmethod
    def _scale_size_based_on_longer_edge(out_size, new_aspect_ratio):
        if new_aspect_ratio < 1:
            new_h = out_size
            new_w = new_h * new_aspect_ratio
        else:
            new_w = out_size
            new_h = new_w / new_aspect_ratio
        return np.array([new_w, new_h]).astype(np.int)


class ToFixedSizeTargets:
    def __init__(self, max_targets):
        self.max_targets = max_targets

    def __call__(self, image, targets=None):
        padded_targets = np.zeros((self.max_targets, 5))
        if targets is not None:
            targets = targets[:self.max_targets]
            padded_targets[:len(targets)] = targets
        return image, padded_targets


class ToTensor:
    def __call__(self, image, targets):
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        if targets is not None:
            targets = torch.from_numpy(targets).float()
        return image, targets


class ToYOLOTargets:
    def __call__(self, image, targets):
        """Convert COCO annotation to target

        Args:
            image (np.ndarray): image
            targets (dict): COCO annotation
                bounding box is represented as (x0, y0, w, h) where
                    (x0, y0): coordinates of top-left corner of it
                    (w, h): size

        Returns:
            (tuple): (image, targets)
                image (np.ndarray): image
                targets (np.ndarray or None): (shape: Nx5)
                    each row is (bbox, category_index) where
                        bbox: (shape: 4) (xc, yc, w, h) where
                            (xc, yc): coordinates of center of the
                                bounding box
                            (w, h): size of the bounding box
                        category_index: category index in the list
                            of categories considered

        """
        converted_targets = []
        for annotation in targets:
            bbox = np.asarray(annotation['bbox'])
            bbox[:2] = bbox[:2] + bbox[2:] / 2
            converted_targets.append([*bbox, annotation['category_index']])
        if len(converted_targets) > 0:
            converted_targets = np.vstack(converted_targets)
        else:
            converted_targets = None
        return image, converted_targets
