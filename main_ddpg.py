#%%
import gymnasium as gym
from gymnasium.spaces import Box

import ray
from ray import tune
from ray.rllib.algorithms.ddpg.ddpg import DDPGConfig, DDPG

from ray.tune.logger import pretty_print
from Environment import *

# gym environment adapter
class SimplePlatform(gym.Env):
    def __init__(self, config):
        self.reset()
        self.action_space = Box(low=0.0, high=1.0, shape=(self.platform.media_size, ), dtype=np.float32)
        self.observation_space = Box(-10000, 10000, shape=(len(self.platform.initial_state().to_array()), ), dtype=np.float32)

    def reset(self):
        self.platform = PlatformEnvironment()
        self.state = self.platform.initial_state()
        return self.state.to_array()

    def step(self, action):
        self.state, reward, done = self.platform.step(self.state, Action(self.platform.media_size))
        return self.state.to_array(), reward, done, {}
    
ray.shutdown()
# ray.init(num_gpus = 1)
ray.init()
def train_ddpg():
    config = DDPGConfig().training(lr=0.01).resources(num_gpus=1)
    #config = DDPGConfig()   ## CPU version
    config["log_level"] = "WARN"
    config["actor_hiddens"] = [512, 512] 
    config["critic_hiddens"] = [512, 512]
    config["gamma"] = 0.99
    config["timesteps_per_iteration"] = 1000
    config["target_network_update_freq"] = 5
    config['replay_buffer_config']['capacity'] = 10000
    
    
    trainer = DDPG(config=config, env=SimplePlatform)
    for i in range(4):
        print(i)
        result = trainer.train()
        print(pretty_print(result))
        checkpoint = trainer.save()
        print("Checkpoint saved at", checkpoint)


train_ddpg()
