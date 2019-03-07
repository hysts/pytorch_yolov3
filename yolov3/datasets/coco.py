import pathlib
import numpy as np
import cv2
import torch.utils.data

from pycocotools.coco import COCO


class COCODataset(torch.utils.data.Dataset):
    def __init__(self,
                 image_dir,
                 annotation_path,
                 is_train,
                 bbox_min_size=0,
                 transform=None):
        """Initialize COCO dataset

        Args:
            image_dir (str): image directory
            annotation_path (str): path to json annotation file
            is_train (bool): whether it's training data or not
            bbox_min_size (int): minimum size of bounding box. This
                argument is used only for training data to remove
                too small objects.
            transform (callable, optional): a function that takes a
                pair of image and targets and return a transformed
                version of it
        """
        self.image_dir = pathlib.Path(image_dir)
        self.coco = COCO(annotation_path)
        self.image_ids = sorted(self.coco.getImgIds())
        self.category_ids = sorted(self.coco.getCatIds())

        self.bbox_min_size = bbox_min_size
        self.is_train = is_train
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        if self.is_train:
            targets = self._preprocess_train_annotations(image_id)
        else:
            targets = None

        path = self.image_dir / f'{image_id:012}.jpg'
        image = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
        h, w = image.shape[:2]
        original_size = np.array([w, h], dtype=np.float32)

        if self.transform is not None:
            image, targets = self.transform(image, targets)

        if self.is_train:
            return image, targets
        else:
            return image, original_size, image_id

    def _preprocess_train_annotations(self, image_id):
        annotation_ids = self.coco.getAnnIds(imgIds=[image_id], iscrowd=None)
        annotations = self.coco.loadAnns(annotation_ids)
        annotations = self._filter_too_small_objects(annotations)
        targets = self._add_category_indices_to_annotations(annotations)
        return targets

    def _filter_too_small_objects(self, annotations):
        filtered_annotations = []
        for annotation in annotations:
            bbox = np.asarray(annotation['bbox'])
            if (bbox[2:] < self.bbox_min_size).any():
                continue
            filtered_annotations.append(annotation)
        return filtered_annotations

    def _add_category_indices_to_annotations(self, annotations):
        for annotation in annotations:
            category_index = self.category_ids.index(annotation['category_id'])
            annotation['category_index'] = category_index
        return annotations
