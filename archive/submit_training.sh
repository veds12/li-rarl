#!/bin/bash
#SBATCH --gres=gpu:rtx8000:1     # Number of GPUs (per node)
#SBATCH --mem=60G        # memory (per node)
#SBATCH --time=04-12:00   # time (DD-HH:MM)

### cluster information above this line

### load environment
module load anaconda/3
module load cuda/10.1
conda activate li-rarl

expt_name=$1

### run training script
python main.py --config config.yaml --suite atari --env atari-breakout --selector attention --summarizer i2a --agent dqn --seed 0 --run test

