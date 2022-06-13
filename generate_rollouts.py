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
CHECKPOINT_PATH = "./ray-rllib/halfcheetah-ppo/PPO_HalfCheetah-v2_167fb_00000_0_2022-05-26_17-51-31/checkpoint_000010/checkpoint-10"
OUTPUT_DIR = "rollouts/RLLIB-PPO-PAP1"

SEED = 1443
N_EPISODES = 100
EPISODE_LENGTH = 1000


def main():
    # Create output directory if it doesn't already exists
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

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

    # Run for the desired number of episodes and steps per episode
    final_dataset_arr = None
    for e in range(N_EPISODES):
        # Reset to begin a new episode
        next_obs = env.reset()

        # Empty starting array
        dataset_arr = None

        # Complete the desired number of steps
        for _ in range(EPISODE_LENGTH):
            obs = np.copy(next_obs)
            action = agent.compute_single_action(obs)
            next_obs, reward, done, _ = env.step(action)
            ep_transition = np.hstack((obs, action, next_obs, np.array((reward)), np.array((int(done)))))

            if dataset_arr is None:
                dataset_arr = np.copy(ep_transition)
            else:
                dataset_arr = np.vstack((dataset_arr, ep_transition))
        
        policies = np.vstack((
            np.full((int(EPISODE_LENGTH/5),1), 0.),
            np.full((int(EPISODE_LENGTH/5),1), 1.),
            np.full((int(EPISODE_LENGTH/5),1), 2.),
            np.full((int(EPISODE_LENGTH/5),1), 3.),
            np.full((int(EPISODE_LENGTH/5),1), 4.),
        ))
        dataset_arr = np.hstack((dataset_arr, policies))
        np.save(os.path.join(OUTPUT_DIR, f'rollout_{EPISODE_LENGTH}_{e}.npy'), dataset_arr)
    
        if final_dataset_arr is None:
            final_dataset_arr = dataset_arr
        else:
            final_dataset_arr = np.vstack((final_dataset_arr, dataset_arr))
    
    # check the dataset size
    assert final_dataset_arr.shape[0] == N_EPISODES*EPISODE_LENGTH
    
    # save final array
    np.save(os.path.join(OUTPUT_DIR, f'combined_data.npy'), final_dataset_arr)

if __name__ == "__main__":
    main()
