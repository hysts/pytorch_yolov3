from yolov3.transforms.transforms import (
    Compose, FlipColorChannelOrder, Normalize, NormalizeTargets, Pad,
    RandomDistort, RandomHorizontalFlip, Resize, ToFixedSizeTargets, ToTensor,
    ToYOLOTargets)


def create_transform(config, is_train):
    if is_train:
        train_transforms = [
            ToYOLOTargets(),
            Resize(config.train.image_size, config.train.augmentation.jitter),
            Pad(config.train.augmentation.random_padding),
            NormalizeTargets(),
        ]
        # RandomHorizontalFlip assumes target to be normalized, so
        # this must be put after NormalizeTargets
        if config.train.augmentation.random_horizontal_flip:
            train_transforms += [RandomHorizontalFlip()]
        if config.train.augmentation.random_distortion:
            train_transforms += [
                RandomDistort(config.train.augmentation.distortion.hue,
                              config.train.augmentation.distortion.saturation,
                              config.train.augmentation.distortion.exposure)
            ]
        # RandomDistort assumes an image has BGR order, so
        # FlipColorChannelOrder must be done after it
        if config.train.channel_order == 'rgb':
            train_transforms += [FlipColorChannelOrder()]

        train_transforms += [
            ToFixedSizeTargets(config.train.max_targets),
        ]
        return Compose(train_transforms)
    else:
        out_size = config.validation.image_size
        val_transforms = [
            Resize(out_size, random_aspect_ratio_jitter=0),
            Pad(random_padding=False),
        ]
        if config.validation.channel_order == 'rgb':
            val_transforms += [FlipColorChannelOrder()]
        val_transforms += [
            Normalize(),
            ToTensor(),
        ]
        return Compose(val_transforms)
