#!/bin/bash


python3 src/main.py --config=qmix --env-config=gymmb with env_args.time_limit=50 env_args.key="lbforaging:Foraging-8x8-2p-3f-v2" 

python3 src/main.py --config=ippo --env-config=gymmb with env_args.time_limit=50 env_args.key="lbforaging:Foraging-8x8-2p-3f-v2" 

python3 src/main.py --config=mappo --env-config=gymmb with env_args.time_limit=50 env_args.key="lbforaging:Foraging-8x8-2p-3f-v2" 

python3 src/main.py --config=iql --env-config=gymmb with env_args.time_limit=50 env_args.key="lbforaging:Foraging-8x8-2p-3f-v2" 
