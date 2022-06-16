#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=07-00:00   # time (DD-HH:MM)
#SBATCH --job-name=train_babyai
#SBATCH --partition=long
#SBATCH --output=/network/scratch/v/vedant.shah/li-rarl/slurms/SPR-%j.out

### cluster information above this line

### load environment
module load anaconda/3
module load cuda/10.1
conda activate li-rarl

seed=$1

### run training script
CUDA_LAUNCH_BLOCKING=1 python train_babyai.py --seed ${seed} --env BabyAI-GoToLocal-v0 --no-instr --arch pixels_endpool_res --name BabyAI-GoToLocal-v0-NoInstr-Retrieval_New --logging --retrieval
# python train_babyai.py --env BabyAI-GoToLocal-v0 --no-instr --arch pixels_endpool_res --name retrieval_debug_32_procs_numpy_buffer_vanilla --retrieval --procs 32 --logging