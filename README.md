# RT-K-Net: Revisiting K-Net for Real-Time Panoptic Segmentation

<a target="_blank">
<img src="/media/cityscapes_demo_video.gif"/>
</a>

This repository contains the official implementation of our IV 2023 paper [RT-K-Net: Revisiting K-Net for Real-Time Panoptic Segmentation](https://arxiv.org/pdf/2305.01255.pdf). 

## Installation

See [INSTALL.md](INSTALL.md) for instructions on how to prepare your environment to use RT-K-Net.

## Usage

See [datasets/README.md](datasets/README.md) for instructions on how to prepare datasets for RT-K-Net.

See [GETTING_STARTED.md](GETTING_STARTED.md) for instructions on how to train and evaluate models, or run inference on demo images.

## Model Zoo

The model files provided below are made available under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.
The Cityscapes model was trained using 4 NVIDIA 2080Ti GPUs, the Mapillary Vistas model was trained using 8 NVIDIA 2080Ti GPUs.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">PQ</th>
<th valign="bottom">PQ_St</th>
<th valign="bottom">PQ_Th</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
 <tr><td align="left"><a href="configs/RT-K-Net-Cityscapes.yaml">RT-K-Net Cityscapes Fine</a></td>
<td align="center">60.2</td>
<td align="center">66.5</td>
<td align="center">51.5</td>
<td align="center"><a href="https://drive.google.com/file/d/1YkoaAMw1lLIckEOYss4qf6EQHKHqoUD2/view?usp=sharing">model</a></td>
</tr>
 <tr><td align="left"><a href="configs/RT-K-Net-Mapillary.yaml">RT-K-Net Mapillary Vistas</a></td>
<td align="center">33.2</td>
<td align="center">45.8</td>
<td align="center">23.6</td>
<td align="center"><a href="https://drive.google.com/file/d/1TF5HULgI9f2jHsItAd7dKNF-JJTRl-Bf/view?usp=sharing">model</a></td>
</tr>
</tbody></table>

## Reference

Please use the following citations when referencing our work:

**RT-K-Net: Revisiting K-Net for Real-Time Panoptic Segmentation (IV 2023)** \
*Markus Schön, Michael Buchholz and Klaus Dietmayer*, [**[arxiv]**](https://arxiv.org/pdf/2305.01255.pdf)

```
@InProceedings{Schoen_2023_IV,
    author    = {Sch{\"o}n, Markus and Buchholz, Michael and Dietmayer, Klaus},
    title     = {RT-K-Net: Revisiting K-Net for Real-Time Panoptic Segmentation},
    booktitle = {IEEE Intelligent Vehicles Symposium},
    year      = {2023}
}
```
## Acknowledgement

We used and modified code parts from other open source projects, we especially like to thank the authors of:

- [detectron2](https://github.com/facebookresearch/detectron2)
- [K-Net](https://github.com/ZwwWayne/K-Net)
- [Mask2Former](https://github.com/facebookresearch/Mask2Former)