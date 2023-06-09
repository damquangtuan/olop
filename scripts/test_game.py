import gym_minigrid as gym
import gymnasium as gym
# env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")
# env = gymnasium.make("highway-v0", render_mode="human")
env = gym.make('FrozenLake-v1', desc=None,map_name="4x4", is_slippery=True)
observation, info = env.reset(seed=42)
for _ in range(1000):
   action = 1 #policy(observation)  # User-defined policy function
   observation, reward, terminated, truncated, info = env.step(action)

   print("reward: " + str(reward))
   if terminated or truncated:
      observation, info = env.reset()
env.close()