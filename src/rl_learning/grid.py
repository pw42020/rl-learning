import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=5):
        super(GridEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32)
        self.state = np.array([0, 0])  # Start at top-left corner
        self.goal = np.array([grid_size-1, grid_size-1])  # Goal at bottom-right corner

    def reset(self, seed=None, options=None):
        self.state = np.array([0, 0])
        return self.state, {}
    
    def step(self, action):
        if action == 0 and self.state[0] > 0:  # Up
            self.state[0] = max(0, self.state[0] - 1)
        elif action == 1 and self.state[0] < self.grid_size - 1:  # Down
            self.state[0] = min(self.grid_size - 1, self.state[0] + 1)
        elif action == 2 and self.state[1] > 0:  # Left
            self.state[1] = max(0, self.state[1] - 1)
        elif action == 3 and self.state[1] < self.grid_size - 1:  # Right
            self.state[1] = min(self.grid_size - 1, self.state[1] + 1)
        
        done = np.array_equal(self.state, self.goal)
        reward = 1 if done else -0.1
        
        return self.state, reward, done, False, {}
    
    def render(self, mode='human'):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        grid[:] = '.'
        grid[self.state[0], self.state[1]] = 'A'  # Agent
        grid[self.goal[0], self.goal[1]] = 'G'    # Goal
        print("\n".join([" ".join(row) for row in grid]))
        print()
    
env = GridEnv(grid_size=5)

# connect to DQN agent and train
from stable_baselines3 import DQN

model = DQN('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000, progress_bar=True)

# Test the trained agent
obs, info = env.reset()
total_reward = 0
done = False
num_iterations = 0
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, truncated, info = env.step(action)
    env.render()
    total_reward += rewards
    num_iterations += 1
    if done:
        print(f"Goal reached after {num_iterations} iterations!")
        break
env.close()

print(f"Total Reward: {total_reward}")
# Save the trained model
model.save("dqn_grid_env")