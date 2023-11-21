# Semantic-guidance-and-spatial-localization-network

## Introduction

This repo is the official implementation of ["Exchanging Dual-Encoder–Decoder: A New Strategy for Change Detection With Semantic Guidance and Spatial Localization"](https://arxiv.org/abs/2311.11302)

## Install dependencies

1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)
2. [Install Pytorch 1.12 or later](https://pytorch.org/get-started/locally/)
3. Install dependencies

​	Use the following code in command line to install dependencies.

`	pip install -r requirements.txt`

## Data

Using any change detection dataset you want, but organize dataset path as follows. `dataset_name`  is name of change detection dataset, you can set whatever you want.

```python
dataset_name
├─train
│  ├─label
│  ├─t1
│  └─t2
├─val
│  ├─label
│  ├─t1
│  └─t2
└─test
    ├─label
    ├─t1
    └─t2
```

Below are some binary change detection dataset you may want.

[WHU Building](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html)

Paper: Fully convolutional networks for multisource building extraction from an open aerial and satellite imagery data set

[DSIFN](https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images/tree/master/dataset)

Paper: A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sensing images

[LEVIR-CD](https://justchenhao.github.io/LEVIR/)

Paper: A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection

[LEVIR-CD+](http://rs.ia.ac.cn/cp/portal/dataDetail?name=LEVIR-CD%2B)

[GoogleMap](https://github.com/daifeng2016/Change-Detection-Dataset-for-High-Resolution-Satellite-Imagery)

Paper: SemiCDNet: A Semisupervised Convolutional Neural Network for Change Detection in High Resolution Remote-Sensing Images

[SYSU-CD](https://hub.fastgit.org/liumency/SYSU-CD)

Paper: SYSU-CD: A new change detection dataset in "A Deeply-supervised Attention Metric-based Network and an Open Aerial Image Dataset for Remote Sensing Change Detection"

[CDD](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9)

Paper: CHANGE DETECTION IN REMOTE SENSING IMAGES USING CONDITIONAL ADVERSARIAL NETWORKS

[NJDS](https://drive.google.com/file/d/1cQRWORIgW-X2BaeRo1hvFj7vlQtwnmne/view?userstoinvite=infinitemabel.wq@gmail.com&ts=636c5f76&actionButton=1&pli=1)

Paper: Semantic feature-constrained multitask siamese network for building change detection in high-spatial-resolution remote sensing imagery

[S2Looking](https://github.com/S2Looking/Dataset)

Paper: S2Looking: A Satellite Side-Looking Dataset for Building Change Detection

## Start

For training, run the following code in command line.

`python train.py`

If you want to debug while training, run the following code in command line.

`python -m ipdb train.py`

For test and inference, run the following code in command line.

`python inference.py` 

## Config

All the configs of dataset, training, validation and test are put in the file "utils/path_hyperparameter.py", you can change the configs in this file.

---

## 简介

这个项目是["Exchanging Dual-Encoder–Decoder: A New Strategy for Change Detection With Semantic Guidance and Spatial Localization"](https://arxiv.org/abs/2311.11302)的官方pytorch实现

## 下载需要的库

1. [下载CUDA](https://developer.nvidia.com/cuda-downloads)
2. [下载1.12或者更新的pytorch](https://pytorch.org/get-started/locally/)
3. 下载其他需要的包

​	在命令行中运行下面的命令下载其他需要的包

`	pip install -r requirements.txt`

## 数据

你可以使用任何你想使用的变化检测数据集，但是文件组织方式需要按照下面的来。`dataset_name`是你设置的变化检测数据集的名字。

```python
dataset_name
├─train
│  ├─label
│  ├─t1
│  └─t2
├─val
│  ├─label
│  ├─t1
│  └─t2
└─test
    ├─label
    ├─t1
    └─t2
```

下面是一些你可能需要的二分类变化检测数据集。

[WHU Building](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html)

Paper: Fully convolutional networks for multisource building extraction from an open aerial and satellite imagery data set

[DSIFN](https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images/tree/master/dataset)

Paper: A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sensing images

[LEVIR-CD](https://justchenhao.github.io/LEVIR/)

Paper: A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection

[LEVIR-CD+](http://rs.ia.ac.cn/cp/portal/dataDetail?name=LEVIR-CD%2B)

[GoogleMap](https://github.com/daifeng2016/Change-Detection-Dataset-for-High-Resolution-Satellite-Imagery)

Paper: SemiCDNet: A Semisupervised Convolutional Neural Network for Change Detection in High Resolution Remote-Sensing Images

[SYSU-CD](https://hub.fastgit.org/liumency/SYSU-CD)

Paper: SYSU-CD: A new change detection dataset in "A Deeply-supervised Attention Metric-based Network and an Open Aerial Image Dataset for Remote Sensing Change Detection"

[CDD](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9)

Paper: CHANGE DETECTION IN REMOTE SENSING IMAGES USING CONDITIONAL ADVERSARIAL NETWORKS

[NJDS](https://drive.google.com/file/d/1cQRWORIgW-X2BaeRo1hvFj7vlQtwnmne/view?userstoinvite=infinitemabel.wq@gmail.com&ts=636c5f76&actionButton=1&pli=1)

Paper: Semantic feature-constrained multitask siamese network for building change detection in high-spatial-resolution remote sensing imagery

[S2Looking](https://github.com/S2Looking/Dataset)

Paper: S2Looking: A Satellite Side-Looking Dataset for Building Change Detection

## 开始

在命令行中运行下面的代码来开始训练

`python train.py`

如果你想在训练的时候进行调试，在命令行中运行下面的命令

`python -m ipdb train.py`

在命令行中运行下面的代码来开始测试或者推理

`python inference.py` 

## 设置

所有和数据集、训练、验证和测试的设置都放在了“utils/path_hyperparameter.py”文件中，你可以在这个文件里修改设置

