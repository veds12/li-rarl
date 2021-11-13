#!/bin/bash
#SBATCH --gres=gpu:1     # Number of GPUs (per node)
#SBATCH --mem=60G        # memory (per node)
#SBATCH --time=0-12:00   # time (DD-HH:MM)

###########cluster information above this line

###load environment
module load cuda/10.0
module load anaconda/3
conda activate lirarl

python dreamer-torch/dreamer.py --logdir ./logs/atari_pong/dreamerv2/1 --configs defaults atari --task atari_pong
