import os

import numpy as np
import gym
import ray
from ray.rllib.agents import ppo, ddpg, sac

########
# Config
########

AGENT = ppo
TRAINER = ppo.PPOTrainer
ENVIRONMENT = "HalfCheetah-v2"
# POLICY_PATH = "ray-rllib/halfcheetah-ppo/PPO_HalfCheetah-v2_ac579_00000_0_2022-05-26_14-35-16"
POLICY_PATH = "ray-rllib/halfcheetah-ppo/PPO_HalfCheetah-v2_167fb_00000_0_2022-05-26_17-51-31"
CHECKPOINT = 1250
CHECKPOINT_PATH = os.path.join(POLICY_PATH, f"checkpoint_{str(CHECKPOINT).rjust(6, '0')}", f"checkpoint-{CHECKPOINT}")

SEED = 1443
EPISODE_LENGTH = 1000


def main():
    # Initialise Ray
    ray.init()

    # Compile configuration
    config = AGENT.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_gpus_per_worker"] = 0
    config["num_workers"] = 1

    # Load gym environment
    env = gym.make(ENVIRONMENT)

    # Set the seed, if provided
    if SEED is not None:
        env.seed(SEED)
        np.random.seed(SEED)

    # Load the agent from checkpoint
    agent = TRAINER(config=config, env=ENVIRONMENT)
    agent.restore(CHECKPOINT_PATH)

    # Reset to begin a new episode
    next_obs = env.reset()

    for _ in range(EPISODE_LENGTH):
        obs = np.copy(next_obs)
        action = agent.compute_single_action(obs)
        next_obs, _, _, _ = env.step(action)
        env.render()


if __name__ == "__main__":
    main()
