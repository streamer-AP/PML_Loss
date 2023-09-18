# Cascade_Loss

## Paper

This is an official code for our paper: Progressive Multi-resolution Loss for Crowd Counting. 

Crowd counting is usually handled in a density map regression fashion, which is supervised via an L2 loss between the predicted density map and ground truth.
To effectively regulate models, various improved L2 loss functions have been developed to find a better correspondence between predicted density and annotation positions.
In this paper, we propose to predict the density map at one resolution but measure its quality via a derived log-formed loss at multiple resolutions. 
Unlike existing methods that assume density maps at different resolutions are independent, 
our loss is obtained by modeling the likelihood function inspired by the relationship of density maps across multi-resolutions.
We find that the traditional single-resolution L2 loss is a particular case of our derived log-likelihood.
We mathematically prove it is superior to a single-resolution L2 loss.
Without bells and whistles, the proposed loss substantially improves several baselines and performs favorably compared to state-of-the-art methods on five crowd counting datasets: NWPU-Crowd, ShanghaiTech A \& B, UCF-QNRF, and JHU-Crowd++.

## Requirements

Put the data in the datasets fold.

This code have been tested on python > 3.8 and pytorch > 2.0, training scripts can be found in scripts: