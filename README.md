# PyTorch YOLOv3

## Introduction

This is a yet another PyTorch implementation of YOLOv3.

With this repository, you can
* run inference with darknet pretrained models
* train with multiple GPUs
* train with COCO-style datasets
* easily change backbone architecture
* reproduce training results (to some extent. still work in progress...)



## Training Results

|                            | Train     | Val       | image size | AP50_95 | AP50  |
|:---------------------------|:----------|:----------|:----------:|:-------:|:-----:|
| Paper                      | COCO 2014 | COCO 2014 |    416     |  31.0   | 55.3  |
| Paper                      | COCO 2014 | COCO 2014 |    608     |  33.0   | 57.9  |
| Converted pretrained model | COCO 2014 | COCO 2014 |    416     |  30.5   | 54.8  |
| Converted pretrained model | COCO 2014 | COCO 2014 |    608     |  32.4   | 57.4  |
| This repository            | COCO 2017 | COCO 2017 |    416     |  30.6   | 52.6  |
| This repository            | COCO 2017 | COCO 2017 |    608     |  31.2   | 53.6  |

For evaluation on COCO 2014, see [this section](#evaluation-using-coco-2014).



## Requirements

* Python 3.6+
* matplotlib
* numpy
* OpenCV
* [pycocotools](https://github.com/cocodataset/cocoapi)
* [tensorboardX](https://github.com/lanpa/tensorboardX) >=1.6
* PyTorch >=1.0.1
* tqdm
* [YACS](https://github.com/rbgirshick/yacs) >=0.1.6



## Installation

### Option 1: Step-by-step installation

```bash
git clone https://github.com/hysts/pytorch_yolov3
cd pytorch_yolov3
pip install cython numpy
python setup.py install
```

### Option 2: Using Docker

You need nvidia-docker2 for this.

#### Build

```bash
docker build . -f docker/Dockerfile -t yolov3
```

#### Run

```bash
docker run --runtime nvidia --ipc host -it yolov3
```



## Inference with pretrained models

### Download darknet pretrained models

Following command downloads pretrained model from
[darknet site](https://pjreddie.com/darknet/yolo/), and convert them to PyTorch
model files this repository uses.

```bash
bash scripts/tools/download_darknet_weights.sh darknet_weights
```

In this repository, images are considered to have BGR channel order,
so we swap channels of the first convolutional layer in darknet pretrained
models, which are trained with images with RGB channel order.

### Run demo

`scripts/demo.py` supports an image input, a video input, and a camera input.
To stop demo, press `q` or `Esc` key on the shown image.

#### On an image

```bash
python scripts/demo.py --ckpt darknet_weights/yolov3.weights.pth --image /path/to/image.jpg
```

#### On an image directory

```bash
python scripts/demo.py --ckpt darknet_weights/yolov3.weights.pth --image_dir /path/to/image/directory
```

To process next image, press any key other than `q` or `Esc`.

#### On a video

```bash
python scripts/demo.py --ckpt darknet_weights/yolov3.weights.pth --video /path/to/video.mp4
```

#### On a webcam

```bash
python scripts/demo.py --ckpt darknet_weights/yolov3.weights.pth --camera
```



## Training

This repository supports training with COCO-style dataset.

### Download COCO 2017 dataset

```bash
bash scripts/tools/download_coco2017.sh ~/datasets/coco2017
```

### Download pretrained backbone model (optional)

You can train YOLOv3 from scratch, but using pretrained backbone weights makes
training faster ([arXiv:1811.08883](https://arxiv.org/abs/1811.08883)).
If you haven't run the script in
[this section](#download-darknet-pretrained-models), run it.

### Write your configuration file

This repository uses [YACS](https://github.com/rbgirshick/yacs) for
configuration.
Default parameters are specified in
[`yolov3/config/defaults.py`](yolov3/config/defaults.py) (which is not
supposed to be modified directly) and you can override them using a YAML file.
All the configurable parameters are listed in
[`configs/yolov3_default.yaml`](configs/yolov3_default.yaml).

You need to create YAML file like that, but you only have to add parameters
you want to override.
For example, if you want to change learning rate,

```yaml
train:
  base_lr: 0.002
```

is enough.

### Training with a single GPU machine

```bash
python scripts/train.py --config /path/to/config/file
```

**Note:** This repository makes it clear not to overwrite output directory
because training object detection model takes so long time that accidentally
corrupting past experiment results is really terrible.
You might find this feature bothersome when you're first trying to run training,
but it's not that bad after all.

### Distributed training with multiple GPUs

This repository supports multi-GPU training with
[`torch.distributed` package](https://pytorch.org/docs/stable/distributed.html).
PyTorch provides `torch.distributed.launch` utility to start multi-GPU
training, and following command is an example to launch training using 8 GPUs.

```bash
python -m torch.distributed.launch --nproc_per_node 8 scripts/train.py --config /path/to/config/file
```

Here, you need to set `train.distributed` in configuration file as follows.

```yaml
train:
  distributed: True
```

**Note:** You could also use `torch.nn.DataParallel` instead of
`torch.nn.parallel.DistributedDataParallel` in this repository, but there's no
reason to use it, as the latter is a lot faster most of the time.

**Note:** Because default parameters specified in
[`yolov3/config/defaults.py`](yolov3/config/defaults.py) are for
single-GPU training, you need to change them in case of multi-GPU training.
If you keep the batch size, you need to raise the learning rate and reduce
the number of iterations. In this case, multiplying learning rate by the number
of GPUs following linear scaling rule
([arXiv:1706.02677](https://arxiv.org/abs/1706.02677)) would be a good idea.
Or, you could divide the batch size by the number of GPUs and reduce
`subdivision` so that `batch_size` / `subdivision` (number of images processed
by each GPU at a time) stays the same.

#### Performance of multi-GPU training

![](figures/multi_gpu_training.png)

As you can see in the figure above, multi-GPU training brings decent speed-up.
Training YOLOv3 takes quite a long time, but with 8 GPUs (Tesla V100),
training would finish in a few days.

### Resume training

You only need to specify the checkpoint directory with `--resume` option.
Configuration file from checkpoint directory is automatically used and even
if you specify configuration file with `--config` option, it's ignored.

#### In case of single GPU training

```bash
python scripts/train.py --resume /path/to/checkpoint/directory
```

#### In case of multi-GPU training

```bash
python -m torch.distributed.launch --nproc_per_node 8 scripts/train.py --resume /path/to/checkpoint/directory
```

### Restart training from a checkpoint, with different configuration

For this, you just need to specify `train.ckpt_path` in a configuration file,
and run as normal.

```yaml
train:
  ckpt_path: /path/to/checkpoint
```



## Evaluation

No validation is run while training in this repository, so you need to validate
separately.
Validation takes following 2 steps.

### Run prediction on validation dataset

This command runs the model on validation dataset, and saves the detection
results to `predictions.json` in the format that can be fed to COCO API.

```bash
python -u scripts/predict.py --config /path/to/config/file \
                             --ckpt_path /path/to/checkpoint \
                             --outdir /path/to/output/directory
```

### Evaluate detection results using COCO API

This command runs evaluation, saves the results to `stats.json`, and also
reports the results to TensorBoard with a step number specified by
`--step` option.

```bash
python -u ./scripts/evaluate_detection_results.py \
    --gt ~/datasets/coco2017/annotations/instances_val2017.json \
    --pred /path/to/predictions.json \
    --step STEP \
    --outdir /path/to/output/directory
```

### Evaluation using COCO 2014

Darknet pretrained models are trained on their own train/val split based on
that of COCO 2014, and that's different from COCO 2017 train/val split.
In order to evaluate darknet pretrained models with this repository, you
need to create JSON version of it that this repository can handle.

Following command downloads the list of validation file names used for darknet
pretrained models, and creates annotation files in JSON format and symbolic
links to COCO 2017 images.
(You need to download COCO 2017 dataset before running this.)

```bash
bash scripts/tools/create_coco2014_json.sh ~/datasets/coco2017 ~/datasets/coco2014
```

Usage of this script is as follows.

```bash
bash scripts/tools/create_coco214_json.sh COCO2017_DIR OUTPUT_DIR
```

The script above generates files like this (images are symbolic links):

```text
OUTPUT_DIR
├── annotations
│   ├── instances_train2014.json
│   └── instances_val2014.json
├── train2014
│   ├── 000000000009.jpg
│   ├── 000000000025.jpg
...
│   └── 000000581929.jpg
└── val2014
    ├── 000000000164.jpg
    ├── 000000000192.jpg
    ...
    └── 000000581899.jpg
```

Using outputs of this script, you can evaluate darknet pretrained models on
COCO 2014  as usual.



## References

* Redmon, Joseph, Santosh Divvala, Ross Girshick, and Ali Farhadi. "You Only Look Once: Unified, Real-Time Object Detection." The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016. [link](http://openaccess.thecvf.com/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html), [arXiv:1506.02640](https://arxiv.org/abs/1506.02640), [Project website](https://pjreddie.com/darknet/yolov1), [GitHub](https://github.com/pjreddie/darknet)
* Redmon, Joseph, and Ali Farhadi. "YOLO9000: Better, Faster, Stronger." The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017. [link](http://openaccess.thecvf.com/content_cvpr_2017/html/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.html), [arXiv:1612.08242](https://arxiv.org/abs/1612.08242), [Project website](https://pjreddie.com/darknet/yolov2), [GitHub](https://github.com/pjreddie/darknet)
* Redmon, Joseph, and Ali Farhadi. "YOLOv3: An Incremental Improvement." arXiv preprint arXiv:1804.02767 (2018). [arXiv:1804.02767](https://arxiv.org/abs/1804.02767), [Project website](https://pjreddie.com/darknet/yolo), [GitHub](https://github.com/pjreddie/darknet)

