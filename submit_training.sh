#!/bin/bash
#SBATCH --gres=gpu:v100:1     # Number of GPUs (per node)
#SBATCH --mem=60G        # memory (per node)
#SBATCH --time=07-00:00   # time (DD-HH:MM)

### cluster information above this line

### load environment
module load anaconda/3
module load cuda/10.1
conda activate li-rarl

### run training script
python main.py --config config.yaml --suite atari --env Atari-Pong --selector kmeans --forward dreamer --agent dqn --seed 43 --run run_1

