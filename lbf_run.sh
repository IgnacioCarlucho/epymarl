#!/bin/bash
#SBATCH --job-name=mpe
#SBATCH --cpus-per-task=4
#SBATCH --time=2-24:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1

# export CUDA_VISIBLE_DEVICES="0"
python3 src/main.py --config=mappo --env-config=gymma with env_args.time_limit=50 env_args.key="lbforaging:MARL-Foraging-12x12-3f-v0" seed=123 descriptor=_mappo
#python3 src/main.py --config=mappo --env-config=gymma with env_args.time_limit=50 env_args.key="lbforaging:MARL-Foraging-12x12-3f-v0" seed=770 descriptor=_mappo
#python3 src/main.py --config=mappo --env-config=gymma with env_args.time_limit=50 env_args.key="lbforaging:MARL-Foraging-12x12-3f-v0" seed=012 descriptor=_mappo
#python3 src/main.py --config=mappo --env-config=gymma with env_args.time_limit=50 env_args.key="lbforaging:MARL-Foraging-12x12-3f-v0" seed=323 descriptor=_mappo
