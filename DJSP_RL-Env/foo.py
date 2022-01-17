
import ray
from ray.tune.registry import register_env
from ray import tune
import gym
from gym import spaces
from gym.utils import EzPickle
import numpy as np

class Foo_Env2(gym.Env, EzPickle):
    def __init__(self, env_config):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=np.float32(0.), high=np.float32(255.), dtype=np.float32, shape=(125,))
        self.cnt = 0

    def step(self, action):
        if action == 3:
            done = True
        elif self.cnt >= 10:
            done = True
        else:
            done = False
        self.cnt += 1
        reward = 1
        return self.current_state(), reward, done, {}

    def reset(self):
        self.cnt = 0
        return self.current_state()
    def current_state(self):
        return np.random.rand(125)

def env_creator(env_config):
    return Foo_Env2(env_config)
def test_ray(env_config):
    ray.init()

    register_env("my_env", env_creator)
    config = {
        "env": "my_env",
        "env_config": env_config,
        "framework": "torch",
        "model": {
            # By default, the MODEL_DEFAULTS dict above will be used.

            # Change individual keys in that dict by overriding them, e.g.
            "fcnet_hiddens": [512, 512, 512],
            "fcnet_activation": "relu",
        },
        "lr": 0.001,

    }
    stop = {
        "training_iteration": 50000,
        "timesteps_total": 50000,
    }

    results = tune.run("PPO", config=config, stop=stop)
    ray.shutdown()

if __name__ == '__main__':
    env_config = {}
    test_ray(env_config)
