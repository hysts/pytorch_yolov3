import pathlib

import torch
import torch.distributed as dist

from yolov3.utils.data.sampler import IterationBasedBatchSampler
from yolov3.utils.data.collator import Collator
from yolov3.datasets import COCODataset
from yolov3.transforms import create_transform


def create_dataset(config, is_train):
    if is_train:
        image_dir = config.data.train_image_dir
        annotation_path = config.data.train_annotation
        bbox_min_size = config.train.bbox_min_size
        transform = create_transform(config, is_train=True)
    else:
        image_dir = config.data.val_image_dir
        annotation_path = config.data.val_annotation
        bbox_min_size = 0
        transform = create_transform(config, is_train=False)

    dataset = COCODataset(
        image_dir=pathlib.Path(image_dir).expanduser(),
        annotation_path=pathlib.Path(annotation_path).expanduser(),
        is_train=is_train,
        bbox_min_size=bbox_min_size,
        transform=transform)
    return dataset


def create_dataloader(config, is_train):
    dataset = create_dataset(config, is_train)

    if is_train:
        if dist.is_available() and dist.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = torch.utils.data.sampler.RandomSampler(
                dataset, replacement=False)
        collator = Collator(
            input_size=config.train.image_size,
            min_size=config.train.augmentation.min_size,
            max_size=config.train.augmentation.max_size)
        train_batch_sampler = IterationBasedBatchSampler(
            sampler,
            batch_size=config.train.batch_size,
            max_iterations=config.train.scheduler.max_iterations,
            drop_last=config.train.dataloader.drop_last,
            start_iter=config.train.start_iter)
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=train_batch_sampler,
            num_workers=config.train.dataloader.num_workers,
            collate_fn=collator,
            pin_memory=config.train.dataloader.pin_memory)
        return train_loader
    else:
        val_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.validation.batch_size,
            num_workers=config.validation.dataloader.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=config.validation.dataloader.pin_memory)
        return val_loader
