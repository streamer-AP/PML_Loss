#!/bin/bash
#SBATCH --gpus=1

module load anaconda/2021.05
source activate torch
NUM_PROC=$1
shift
python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --use_env main.py "$@"

