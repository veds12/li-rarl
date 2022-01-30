#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=0-07:00:00
#SBATCH --job-name=train_dqn
#SBATCH --partition=main
#SBATCH --output=/network/scratch/v/vedant.shah/slurms/DQN-Pong-%j.out


### cluster information above this line

### load environment
module load anaconda/3
module load cuda/10.1
conda activate li-rarl

seed=$1

### run training script
CUDA_LAUNCH_BLOCKING=1 python train_atari.py --seed ${seed} --logging 1
