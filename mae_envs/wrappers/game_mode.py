import gym
import numpy as np
from copy import deepcopy
from mae_envs.wrappers.util import update_obs_space


class GameMode(gym.ObservationWrapper):
    '''
        Gamemode based on playground and mode.
    '''
    def __init__(self, env, playground, mode):
        super().__init__(env)
        self.n_agents = self.metadata['n_agents']
        self.playground = playground
        self.mode = mode
        self.observation_space = update_obs_space(self, {'game_mode': [self.n_agents, 2]})

    def observation(self, obs):
        obs['game_mode'] = np.array((np.ones(self.n_agents) * self.playground,
                                    np.ones(self.n_agents) * self.mode)).transpose()
        
        return obs
