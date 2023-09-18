#!/bin/bash
#SBATCH --gpus=1

module load anaconda/2021.05
source activate torch
date
NUM_PROC=$1
shift

python train_locater.py "$@"
