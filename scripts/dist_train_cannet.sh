#!/bin/bash
#SBATCH --gpus=1

module load anaconda/2021.05
source activate torch
date
NUM_PROC=$1
shift

torchrun --master_port 29646 --nproc_per_node=$NUM_PROC train_cannet.py "$@"
