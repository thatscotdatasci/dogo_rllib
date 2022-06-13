import os
from collections import namedtuple

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
OUTPUT_DIR = "rollouts/RLLIB-PPO-MP1"

SEED = 1443

POLICY_PATH = "/home/ajc348/rds/hpc-work/dogo_rllib/ray-rllib/halfcheetah-ppo/PPO_HalfCheetah-v2_167fb_00000_0_2022-05-26_17-51-31"
RecipeIngredient = namedtuple('RecipeIngredient', 'checkpoint n_episodes episode_length')
RECIPE = [
    RecipeIngredient(250,  20,  1000),
    RecipeIngredient(750,  20,  1000),
    RecipeIngredient(1250, 20,  1000),
    RecipeIngredient(1750, 20,  1000),
    RecipeIngredient(2250, 20,  1000),
]

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

    final_dataset_arr = None
    for i, ing in enumerate(RECIPE):
        checkpoint_path = os.path.join(POLICY_PATH, f"checkpoint_{str(ing.checkpoint).rjust(6, '0')}", f"checkpoint-{ing.checkpoint}")

        # Load the agent from checkpoint
        agent = TRAINER(config=config, env=ENVIRONMENT)
        agent.restore(checkpoint_path)

        # Run for the desired number of episodes and steps per episode
        ingredient_dataset_arr = None
        for e in range(ing.n_episodes):
            # Reset to begin a new episode
            next_obs = env.reset()

            # Empty starting array
            dataset_arr = None

            # Complete the desired number of steps
            for _ in range(ing.episode_length):
                obs = np.copy(next_obs)
                action = agent.compute_single_action(obs)
                next_obs, reward, done, _ = env.step(action)
                policy = np.array((float(i)))
                ep_transition = np.hstack((obs, action, next_obs, np.array((reward)), np.array((int(done))), policy))

                if dataset_arr is None:
                    dataset_arr = np.copy(ep_transition)
                else:
                    dataset_arr = np.vstack((dataset_arr, ep_transition))
            
            np.save(os.path.join(OUTPUT_DIR, f'rollout_{i}_{ing.episode_length}_{e}.npy'), dataset_arr)
        
            if ingredient_dataset_arr is None:
                ingredient_dataset_arr = dataset_arr
            else:
                ingredient_dataset_arr = np.vstack((ingredient_dataset_arr, dataset_arr))
    
        # check the dataset size
        assert ingredient_dataset_arr.shape[0] == ing.n_episodes*ing.episode_length

        if final_dataset_arr is None:
            final_dataset_arr = ingredient_dataset_arr
        else:
            final_dataset_arr = np.vstack((final_dataset_arr, ingredient_dataset_arr))
    
    # save final array
    np.save(os.path.join(OUTPUT_DIR, f'combined_data.npy'), final_dataset_arr)

if __name__ == "__main__":
    main()
