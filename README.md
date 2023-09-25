# PML Loss

# Paper

This is an official code for our paper: Progressive Multi-resolution Loss for Crowd Counting. 

Crowd counting is usually handled in a density map regression fashion, which is supervised via an L2 loss between the predicted density map and ground truth.
To effectively regulate models, various improved L2 loss functions have been developed to find a better correspondence between predicted density and annotation positions.
In this paper, we propose to predict the density map at one resolution but measure its quality via a derived log-formed loss at multiple resolutions. 
Unlike existing methods that assume density maps at different resolutions are independent, 
our loss is obtained by modeling the likelihood function inspired by the relationship of density maps across multi-resolutions.
We find that the traditional single-resolution L2 loss is a particular case of our derived log-likelihood.
We mathematically prove it is superior to a single-resolution L2 loss.
Without bells and whistles, the proposed loss substantially improves several baselines and performs favorably compared to state-of-the-art methods on five crowd counting datasets: NWPU-Crowd, ShanghaiTech A \& B, UCF-QNRF, and JHU-Crowd++.

# Prerequisites

Python >=3.8

Pytorch >= 2.0

For other libraries, check requirements.txt.


# Getting Started
## Dataset download

+ UCF-QNRF can be downloaded [here](https://www.crcv.ucf.edu/data/ucf-qnrf/)

+ NWPU-Crowd can be downloaded [here](https://www.crowdbenchmark.com/nwpucrowd.html)

+ JHU-Crowd++ can be downloaded [here](http://www.crowd-counting.com/)

+ Shanghai Tech Part A and Part B can be downloaded [here](https://www.kaggle.com/tthien/shanghaitech)

## Data preprocess

Convert all dataset annotations to COCO format.

+ Run ```./datasets/nwpu2coco.py ``` to generate annotations for the NWPU-Crowd and JHU-Crowd++.

+ Run ```./datasets/st2coco.py ``` to generate annotations for the Shanghai Tech Part A and Part B.

+ Run ```./datasets/ucf2coco.py ``` to generate annotations for UCF-QNRF.

+ Modify  ```./configs/example.json ``` with your dataset paths and COCO format annotation paths.

## Training 

+ run ```python train_csr.py```.

## Pre-trained Models

We provide the pre-trained models in this [link]().

