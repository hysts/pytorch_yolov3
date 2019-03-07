#!/usr/bin/env python

import argparse
import pathlib
import random

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
from tensorboardX import SummaryWriter

import yolov3.models
from yolov3.config import get_default_config
from yolov3.utils.checkpoint import CheckPointer
from yolov3.utils.data.dataloader import create_dataloader
from yolov3.utils.logger import create_logger
from yolov3.utils.scheduler import create_scheduler


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    if args.config is not None:
        config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    if not torch.cuda.is_available():
        config.train.device = 'cpu'
        config.train.dataloader.pin_memory = False
    if args.resume != '':
        config_path = pathlib.Path(args.resume) / 'config.yaml'
        config.merge_from_file(config_path.as_posix())
        config.merge_from_list(['train.resume', True])
    config.merge_from_list(['train.dist.local_rank', args.local_rank])
    config.freeze()
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def create_model(config):
    local_rank = config.train.dist.local_rank
    device = torch.device(config.train.device, local_rank)

    model = yolov3.models.YOLOv3(config)
    model.to(device)
    if dist.is_available() and dist.is_initialized():
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank)
    else:
        if config.train.device == 'cuda':
            model = nn.DataParallel(model)
    return model


def create_optimizer(model, config):
    param_list = []
    for name, params in model.named_parameters():
        if 'conv.weight' in name:
            param_list.append({
                'params': params,
                'weight_decay': config.train.weight_decay,
            })
        else:
            param_list.append({
                'params': params,
                'weight_decay': 0,
            })
    optimizer = torch.optim.SGD(
        param_list,
        lr=config.train.base_lr,
        momentum=config.train.momentum,
        nesterov=config.train.nesterov)
    return optimizer


def get_rank():
    if not (dist.is_available() and dist.is_initialized()):
        return 0
    else:
        return dist.get_rank()


def save_config(config, outdir):
    with open(outdir / 'config.yaml', 'w') as fout:
        fout.write(str(config))


def main():
    config = load_config()

    torch.backends.cudnn.benchmark = (config.train.augmentation.min_size ==
                                      config.train.augmentation.max_size)

    set_seed(config.train.seed)

    if config.train.distributed:
        dist.init_process_group(
            backend=config.train.dist.backend,
            init_method=config.train.dist.init_method,
            rank=config.train.dist.node_rank,
            world_size=config.train.dist.world_size)

    outdir = pathlib.Path(config.train.outdir)
    if get_rank() == 0:
        if not config.train.resume and outdir.exists():
            raise RuntimeError(
                f'Output directory `{outdir.as_posix()}` already exists')
        outdir.mkdir(exist_ok=True, parents=True)
        if not config.train.resume:
            save_config(config, outdir)

    logger = create_logger(
        name=__name__,
        outdir=outdir,
        distributed_rank=get_rank(),
        filename='log.txt')
    logger.info(config)

    model = create_model(config)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    checkpointer = CheckPointer(
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir=outdir,
        logger=logger,
        distributed_rank=get_rank())

    start_iter = config.train.start_iter
    scheduler.last_epoch = start_iter
    if config.train.resume:
        ckpt_config, start_iter = checkpointer.load()
        config.defrost()
        config.merge_from_other_cfg(ckpt_config)
        config.train.start_iter = start_iter
        config.freeze()
    elif config.train.ckpt_path != '':
        _, start_iter = checkpointer.load(
            config.train.ckpt_path, backbone=False)
        config.defrost()
        config.train.start_iter = start_iter
        config.freeze()
        for index in range(len(scheduler.base_lrs)):
            scheduler.base_lrs[index] = config.train.base_lr
        save_config(config, outdir)
    elif config.train.backbone_weight != '':
        checkpointer.load(config.train.backbone_weight, backbone=True)

    if get_rank() == 0 and config.train.tensorboard:
        if start_iter > 0:
            writer = SummaryWriter(
                outdir.as_posix(), purge_step=start_iter + 1)
        else:
            writer = SummaryWriter(outdir.as_posix())
    else:
        writer = None

    train_loader = create_dataloader(config, is_train=True)

    for step, (data, targets) in enumerate(train_loader, start_iter):
        step += 1

        scheduler.step()
        data_chunks = data.chunk(config.train.subdivision)
        target_chunks = targets.chunk(config.train.subdivision)

        optimizer.zero_grad()
        for data_chunk, target_chunk in zip(data_chunks, target_chunks):
            losses = model(data_chunk, target_chunk)
            losses[0].backward()
        if config.train.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           config.train.gradient_clip)
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.div_(data.size(0))
        optimizer.step()

        losses *= config.train.subdivision / config.train.batch_size
        if step == 1 or step % config.train.log_period == 0:
            logger.info(f'iter {step}/'
                        f'{config.train.scheduler.max_iterations}, '
                        f'lr {scheduler.get_lr()[0]:.7f}, '
                        f'image size: {data.shape[2]}, '
                        f'losses: [total {losses[0].item():.3f}, '
                        f'xy {losses[1].item():.3f}, '
                        f'wh {losses[2].item():.3f}, '
                        f'object {losses[3].item():.3f}, '
                        f'class {losses[4].item():.3f}]')

        if writer is not None:
            writer.add_scalar('train/loss', losses[0], step)
            writer.add_scalar('train/loss_xy', losses[1], step)
            writer.add_scalar('train/loss_wh', losses[2], step)
            writer.add_scalar('train/loss_object', losses[3], step)
            writer.add_scalar('train/loss_class', losses[4], step)
            writer.add_scalar('train/lr', scheduler.get_lr()[0], step)
            writer.add_scalar('train/image_size', data.shape[2], step)

        if (step % config.train.ckpt_period == 0) or (
                step == config.train.scheduler.max_iterations):
            ckpt_config = {'iteration': step, 'config': config}
            checkpointer.save(f'model_{step:07d}', **ckpt_config)

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()
