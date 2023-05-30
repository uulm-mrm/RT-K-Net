# Getting Started

This document provides a brief introduction on how to use RT-K-Net.

## Inference Demo with Pre-trained Models

1. Pick a model and its config file, for example, configs/RT-K-Net-Cityscapes.yaml. 
2. We provide demo.py that is able to demo builtin configs. Run it with:

```shell
python tools/demo.py --config-file /path/to/config_file --input /path/to/image_file --opts MODEL.WEIGHTS /path/to/checkpoint_file
```

The configs are made for training, therefore we need to specify MODEL.WEIGHTS to a model from model zoo for evaluation. This command will run the inference and show visualizations.

For details of the command line arguments, see `python tools/demo.py -h` or look at its source code
to understand its behavior. Some common arguments are:
* To run on a video, replace `--input files` with `--video-input video.mp4`.
* To save outputs to a directory (for images) or a file (for video), use `--output`.

If you use docker and get the error that matplotlib cannot render images, you might have to enable root access to the XServer by running `xhost +local:root`.

## Training & Evaluation in Command Line

### Training

We provide a script `train_net.py`, that is made to train all the configs provided.

To train a model with "train_net.py", first setup the corresponding datasets following [datasets/README.md](./datasets/README.md), then run:
```shell
python tools/train_net.py --num-gpus <number_of_gpus> --config-file /path/to/config_file
```
The Cityscapes config is configured for 4 GPU training, the Mapillary Vistas config is configured for 8 GPU training. We use NVIDIA RTX 2080Ti GPUs.

### Evaluation

To evaluate a model's performance, use
```shell
python tools/train_net.py --num-gpus 1 --config-file /path/to/config_file --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```
For more options, see `python tools/train_net.py -h`.

## Data Visualization

We provide a script `visualize_data.py`, that is made to visualize data loaded into the model during training time. 
Hence, images and labels will be visualized with training augmentations. This is useful e.g. to inspect the quality of labels if training on a custom dataset.

To run the script, use
```shell
python tools/visualize_data.py --config-file /path/to/config_file
```