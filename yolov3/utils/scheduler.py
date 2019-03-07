import bisect
import functools
import numpy as np
import torch


def linear_warmup(step, total_steps, lr_start_factor):
    return lr_start_factor + (1 - lr_start_factor) * step / total_steps


def exponential_warmup(step, total_steps, lr_start_factor, exponent):
    return lr_start_factor + (1 - lr_start_factor) * (
        step / total_steps)**exponent


def multistep_decay(step, lr_decay, milestones):
    return lr_decay**bisect.bisect_right(milestones, step)


def linear_decay(step, total_steps, lr_min_factor):
    return 1 - (1 - lr_min_factor) * step / total_steps


def cosine_decay(step, total_steps, lr_min_factor):
    return lr_min_factor + (1 - lr_min_factor) * 0.5 * (
        1 + np.cos(step / total_steps * np.pi))


def create_scheduler(optimizer, config):
    scheduler_config = config.train.scheduler
    warmup_type = scheduler_config.warmup_type
    scheduler_type = scheduler_config.scheduler_type
    warmup_steps = scheduler_config.warmup_steps
    warmup_start_factor = scheduler_config.warmup_start_factor

    if warmup_type == 'linear':
        warmup_scheduler = functools.partial(
            linear_warmup,
            total_steps=warmup_steps,
            lr_start_factor=warmup_start_factor,
        )
    elif warmup_type == 'exponential':
        warmup_scheduler = functools.partial(
            exponential_warmup,
            total_steps=warmup_steps,
            lr_start_factor=warmup_start_factor,
            exponent=scheduler_config.exponent,
        )

    if scheduler_type == 'multistep':
        milestones = scheduler_config.milestones
        lr_decay = scheduler_config.lr_decay
        milestones = np.asarray(milestones) - warmup_steps
        main_scheduler = functools.partial(
            multistep_decay,
            lr_decay=lr_decay,
            milestones=milestones,
        )
    elif scheduler_type == 'linear':
        main_scheduler = functools.partial(
            linear_decay,
            total_steps=scheduler_config.max_iterations - warmup_steps,
            lr_min_factor=scheduler_config.lr_min_factor,
        )
    elif scheduler_type == 'cosine':
        main_scheduler = functools.partial(
            cosine_decay,
            total_steps=scheduler_config.max_iterations - warmup_steps,
            lr_min_factor=scheduler_config.lr_min_factor,
        )

    def combined_scheduler(step):
        if step < warmup_steps:
            return warmup_scheduler(step)
        else:
            return main_scheduler(step - warmup_steps)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: combined_scheduler(step))

    return scheduler
