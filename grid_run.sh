#!/bin/bash


python3 src/main.py --config=ippo --env-config=gymmb with env_args.time_limit=200 env_args.key="grid_generalization:grid-generalization-2p-1w-v0" seed=4321
python3 src/main.py --config=ippo --env-config=gymmb with env_args.time_limit=200 env_args.key="grid_generalization:grid-generalization-2p-2w-v0" seed=4321
python3 src/main.py --config=ippo --env-config=gymmb with env_args.time_limit=200 env_args.key="grid_generalization:grid-generalization-2p-3w-v0" seed=4321
python3 src/main.py --config=ippo --env-config=gymmb with env_args.time_limit=200 env_args.key="grid_generalization:grid-generalization-2p-4w-v0" seed=4321
python3 src/main.py --config=ippo --env-config=gymmb with env_args.time_limit=200 env_args.key="grid_generalization:grid-generalization-2p-5w-v0" seed=4321