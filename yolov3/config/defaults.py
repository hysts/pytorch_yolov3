from yolov3.utils.config_node import ConfigNode

# model
config = ConfigNode()
config.model = ConfigNode()
config.model.name = 'yolov3'
config.model.backbone = 'darknet53'
config.model.anchors = [
    [10, 13],
    [16, 30],
    [33, 23],
    [30, 61],
    [62, 45],
    [59, 119],
    [116, 90],
    [156, 198],
    [373, 326],
]
config.model.anchor_indices = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

# data
config.data = ConfigNode()
config.data.n_classes = 80
dataset_dir = '~/datasets/coco2017'
annotation_dir = f'{dataset_dir}/annotations'
config.data.train_image_dir = f'{dataset_dir}/train2017'
config.data.val_image_dir = f'{dataset_dir}/val2017'
config.data.train_annotation = f'{annotation_dir}/instances_train2017.json'
config.data.val_annotation = f'{annotation_dir}/instances_val2017.json'

# train
config.train = ConfigNode()
config.train.backbone_weight = ''
config.train.ckpt_path = ''
config.train.resume = False
config.train.device = 'cuda'
config.train.image_size = 608
config.train.channel_order = 'bgr'
config.train.batch_size = 64
config.train.subdivision = 8
config.train.optimizer = 'sgd'
config.train.base_lr = 1e-3
config.train.momentum = 0.9
config.train.nesterov = False
config.train.weight_decay = 5e-4
config.train.gradient_clip = 0
config.train.bbox_min_size = 2
config.train.max_targets = 50
config.train.iou_thresh = 0.7
config.train.start_iter = 0
config.train.seed = 0

config.train.outdir = 'results'
config.train.log_period = 10
config.train.ckpt_period = 5000
config.train.tensorboard = True

# scheduler
config.train.scheduler = ConfigNode()
config.train.scheduler.max_iterations = 500000
config.train.scheduler.warmup_type = 'exponential'
config.train.scheduler.exponent = 4
config.train.scheduler.scheduler_type = 'multistep'
config.train.scheduler.warmup_steps = 1000
config.train.scheduler.warmup_start_factor = 1e-3
config.train.scheduler.milestones = [400000, 450000]
config.train.scheduler.lr_decay = 0.1
config.train.scheduler.lr_min_factor = 0.001

# augmentation
config.train.augmentation = ConfigNode()
config.train.augmentation.min_size = 320
config.train.augmentation.max_size = 608
config.train.augmentation.random_horizontal_flip = True
config.train.augmentation.jitter = 0.3
config.train.augmentation.random_padding = True
config.train.augmentation.random_distortion = True
config.train.augmentation.distortion = ConfigNode()
config.train.augmentation.distortion.hue = 0.1
config.train.augmentation.distortion.saturation = 1.5
config.train.augmentation.distortion.exposure = 1.5

# train data loader
config.train.dataloader = ConfigNode()
config.train.dataloader.num_workers = 2
config.train.dataloader.drop_last = True
config.train.dataloader.pin_memory = False

# distributed
config.train.distributed = False
config.train.dist = ConfigNode()
config.train.dist.backend = 'nccl'
config.train.dist.init_method = 'env://'
config.train.dist.world_size = -1
config.train.dist.node_rank = -1
config.train.dist.local_rank = 0

# validation
config.validation = ConfigNode()
config.validation.ckpt_path = ''
config.validation.device = 'cuda'
config.validation.image_size = 416
config.validation.channel_order = 'bgr'
config.validation.batch_size = 128
config.validation.conf_thresh = 0.005
config.validation.nms_thresh = 0.45
config.validation.outdir = ''

# validation data loader
config.validation.dataloader = ConfigNode()
config.validation.dataloader.num_workers = 2
config.validation.dataloader.pin_memory = False

# demo
config.demo = ConfigNode()
config.demo.ckpt_path = ''
config.demo.device = 'cuda'
config.demo.image_size = 416
config.demo.channel_order = 'bgr'
config.demo.conf_thresh = 0.8
config.demo.nms_thresh = 0.45
config.demo.category_names = 'data/coco_names.json'
config.demo.image_path = ''
config.demo.image_dir = ''
config.demo.video_path = ''
config.demo.camera = False


def get_default_config():
    return config.clone()
