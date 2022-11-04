#!/bin/bash
#SBATCH --job-name=cook_qmix
#SBATCH --cpus-per-task=4
#SBATCH --time=8-24:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1

source activate epymarl

export OMP_NUM_THREADS=1 

# export CUDA_VISIBLE_DEVICES="0"

# seeds 1234, 55, 753
python3 src/main.py --config=qmix --env-config=gymmc with env_args.time_limit=50 env_args.key="gym_cooking:cookingZooEnv-tiny-250-v0"  seed=1234