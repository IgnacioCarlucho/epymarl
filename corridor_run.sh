#!/bin/bash

python3 src/main.py --config=qmix --env-config=corridor2d with env_args.time_limit=60 env_args.key="coopreaching:MARL-CooperativeReaching-5-40-v0"  seed=1234

# python3 src/main.py --config=ippo --env-config=corridor2d with env_args.time_limit=50 env_args.key="coopreaching:MARL-CooperativeReaching-5-40-v0" seed=1234

# python3 src/main.py --config=mappo --env-config=corridor2d with env_args.time_limit=50 env_args.key="coopreaching:MARL-CooperativeReaching-5-40-v0" seed=1234