#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=0-07:00:00
#SBATCH --job-name=dqn
#SBATCH --partition=main

### cluster information above this line

### load environment
module load anaconda/3
module load cuda/10.1
conda activate li-rarl

### run training script
python train_atari.py
