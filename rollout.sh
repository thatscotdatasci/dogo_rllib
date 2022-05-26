#! /bin/bash

# Run an episode
# NOTE: This does not seem to work - however render_rollout.py and generate_rollouts.py both do work
rllib evaluate \
    ./ray-rllib/halfcheetah-ppo/PPO_HalfCheetah-v2_ac579_00000_0_2022-05-26_14-35-16/checkpoint_000010/checkpoint-10 \
    --run PPO \
    --env HalfCheetah-v2 \
    --steps 1000 \
    --out rollouts.pkl \
    --config '{"num_gpus": 0, "num_gpus_per_worker": 0}' \
    --render
