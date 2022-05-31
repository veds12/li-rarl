#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=0-01:25   # time (DD-HH:MM)
#SBATCH --job-name=train_babyai


### cluster information above this line

### load environment
module load anaconda/3
module load cuda/10.1
conda activate li-rarl

seed=$1

### run training script
CUDA_LAUNCH_BLOCKING=1 python train_babyai.py --env BabyAI-GoToLocal-v0 --no-instr --arch pixels_endpool_res --retrieval --name BabyAI-GoToLocal-v0-NoInstr-Retrieval --logging