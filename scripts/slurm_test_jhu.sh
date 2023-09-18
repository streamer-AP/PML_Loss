#!/bin/bash
#SBATCH --gpus=1

module load anaconda/2021.05
source activate torch
date
NUM_PROC=$1
shift
torchrun --master_port 30032  --nproc_per_node=$NUM_PROC  test_jhu_model.py"$@"
