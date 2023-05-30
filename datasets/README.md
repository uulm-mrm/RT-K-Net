# Prepare Datasets for RT-K-Net

A dataset can be used by accessing [DatasetCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.DatasetCatalog)
for its data, or [MetadataCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.MetadataCatalog) for its metadata (class names, etc).
This document explains how to setup the builtin datasets so they can be used by the above APIs.
[Use Custom Datasets](https://detectron2.readthedocs.io/tutorials/datasets.html) gives a deeper dive on how to use `DatasetCatalog` and `MetadataCatalog`,
and how to add new datasets to them.

RT-K-Net has builtin support for two datasets.
The datasets are assumed to exist in a directory specified by the environment variable
`DETECTRON2_DATASETS`.
Under this directory, detectron2 will look for datasets in the structure described below, if needed.
```
$DETECTRON2_DATASETS/
  cityscapes/
  mapillary_vistas/
```

You can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`.
If left unset, the default is `./datasets` relative to your current working directory.


## Expected dataset structure for [cityscapes](https://www.cityscapes-dataset.com/downloads/):
Download `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip` from the [Cityscapes](https://www.cityscapes-dataset.com/downloads/) website and extract them to match the following structure:
```
cityscapes/
  gtFine/
    train/
      aachen/
        color.png, instanceIds.png, labelIds.png, polygons.json,
        labelTrainIds.png
      ...
    val/
    test/
    # below are generated Cityscapes panoptic annotation
    cityscapes_panoptic_train.json
    cityscapes_panoptic_train/
    cityscapes_panoptic_val.json
    cityscapes_panoptic_val/
    cityscapes_panoptic_test.json
    cityscapes_panoptic_test/
  leftImg8bit/
    train/
    val/
    test/
```
Install cityscapes scripts by:
```
git clone https://github.com/mcordts/cityscapesScripts.git
cd cityscapesScripts
pip install git+https://github.com/mcordts/cityscapesScripts.git
```

Run the following to create labelTrainIds.png and the Cityscapes panoptic dataset:
```
CITYSCAPES_DATASET=/path/to/abovementioned/cityscapes python cityscapesscripts/preparation/createTrainIdLabelImgs.py
CITYSCAPES_DATASET=/path/to/abovementioned/cityscapes python cityscapesscripts/preparation/createPanopticImgs.py
```


## Expected dataset structure for [Mapillary Vistas](https://www.mapillary.com/dataset/vistas):
Download the dataset from the [Mapillary Vistas](https://www.mapillary.com/dataset/vistas) website and extract it to match the following structure:

```
mapillary_vistas/
  training/
    images/
    instances/
    labels/
    panoptic/
  validation/
    images/
    instances/
    labels/
    panoptic/
```

No preprocessing is needed for Mapillary Vistas on panoptic segmentation.

Note: we use version 1.2 of the Mapillary Vistas dataset.
