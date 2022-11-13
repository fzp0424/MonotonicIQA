# MonotonicIQA

The code for [[2209.10451\] Learning from Mixed Datasets: A Monotonic Image Quality Assessment Model (arxiv.org)](https://arxiv.org/abs/2209.10451)

![Fig2](C:\Users\pp\Desktop\IET_Final\Fig2.png)

# Requirements:

Python 3+
PyTorch 1.4+
Matlab
Successfully tested on Ubuntu 20.04

# Usage

## Sampling images from each datasets(Matlab)

sample_name.m

## Mixing all the sampled images (Matlab)

combine_pmtrain.m

##Train on the mixed datasets for 10 sessions

python Main.py --train True --network basecnn --representation BCNN --batch_size 32 --image_size 384 --lr 3e-4 --decay_interval 3 --decay_ratio 0.9 --max_epochs 24 --backbone resnet34

## Get scores

python Main.py --train False --get_scores True

## Result analysis

Compute weighted PLCC/SRCC: calculate_mean.m

# Acknowledgement

[zwx8981/UNIQUE: The repository for 'Uncertainty-aware blind image quality assessment in the laboratory and wild' and 'Learning to blindly assess image quality in the laboratory and wild' (github.com)](https://github.com/zwx8981/UNIQUE)
