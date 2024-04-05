import gym
import numpy as np
from copy import deepcopy
from functools import partial
from mae_envs.envs.base import Base
from mae_envs.wrappers.multi_agent import (SplitMultiAgentActions,
                                           SplitObservations, SelectKeysWrapper)
from mae_envs.wrappers.util import (DiscretizeActionWrapper, ConcatenateObsWrapper,
                                    MaskActionWrapper, SpoofEntityWrapper,
                                    DiscardMujocoExceptionEpisodes)
from mae_envs.wrappers.manipulation import (GrabObjWrapper, GrabClosestWrapper,
                                            LockObjWrapper, LockAllWrapper)
from mae_envs.wrappers.lidar import Lidar
from mae_envs.wrappers.line_of_sight import AgentAgentObsMask2D, AgentGeomObsMask2D
from mae_envs.wrappers.team import TeamMembership
from mae_envs.modules.agents import Agents, AgentManipulation
from mae_envs.modules.walls import Wall, WallScenarios
from mae_envs.modules.objects import Boxes, Ramps, Cylinders, LidarSites
from mae_envs.modules.world import FloorAttributes, WorldConstants
from mae_envs.modules.util import uniform_placement


'''
Sets up the game environment.
'''
class GameEnvironment:
    def __init__(self, playground, mode, floor_size, grid_size, lidar):
        # Display constants
        self.team_colors = [[np.array((66., 235., 244., 255.)) / 255], [(1., 0., 0., 1.)]]
        self.horizon = 80
        self.visualize_lidar = False

        # Policy constants
        self.substeps = 15
        self.action_lims = (-0.9, 0.9)
        self.deterministic = False
        self.preparation_time = 0.4
        self.reward_type = 'joint_zero_sum'

        # Playground constants
        self.scenario = GameScenario(playground, mode, floor_size, grid_size, lidar)
        self.playground_constants = self.scenario.get_playground_constants()
        for key, value in self.playground_constants.items(): 
            setattr(self, key, value)
        self.walls = self.scenario.get_walls()
        self.agents = self.scenario.get_agents()
        self.boxes = self.scenario.get_boxes()
        self.ramps = self.scenario.get_ramps()
        self.flags = self.scenario.get_flags()
        self.reward_fn = self.scenario.get_reward()

    def construct_env(self):
        # Create base environment
        env = Base(n_agents = len(self.agents[0] + self.agents[1]),
                   n_substeps = self.substeps,
                   horizon = self.horizon,
                   floor_size = self.floor_size,
                   grid_size = self.grid_size,
                   action_lims = self.action_lims,
                   deterministic_mode = self.deterministic)

        # Add walls to the environment
        walls = WallScenarios(grid_size = self.grid_size,
                              walls = self.walls[0],
                              walls_to_split = self.walls[1],
                              door_size = self.door_size,
                              friction = self.object_friction)
        env.add_module(walls)

        # Add agents to the environment
        agents = Agents(n_agents = len(self.agents[0] + self.agents[1]),
                        placement_fn = self.agents[0] + self.agents[1],
                        color = self.team_colors[0] * len(self.agents[0]) + self.team_colors[1] * len(self.agents[1]),
                        friction = self.object_friction)
        env.add_module(agents)
        env.add_module(AgentManipulation())

        # Add boxes to the environment
        boxes = Boxes(n_boxes = len(self.boxes[0] + self.boxes[1]),
                      n_elongated_boxes = len(self.boxes[1]),
                      placement_fn = self.boxes[1] + self.boxes[0],
                      box_size = self.box_size,
                      box_length = self.box_length,
                      box_width = self.box_width,
                      box_height = self.box_height,
                      friction = self.floor_friction,
                      boxid_obs = False,
                      box_only_z_rot = True,
                      alignment = self.alignment * len(self.boxes[1]))
        env.add_module(boxes)

        # Add ramps to the environment
        ramps = Ramps(n_ramps = len(self.ramps),
                      placement_fn = self.ramps,
                      friction = self.object_friction,
                      pad_ramp_size = True)
        env.add_module(ramps)

        # Add flags and zones to the environment
        flags = Cylinders(n_objects = len(self.flags[0] + self.flags[1]),
                          diameter = [self.flag_diameter] * len(self.flags[0]) + [self.zone_diameter] * len(self.flags[1]),
                          height = [self.flag_height] * len(self.flags[0]) + [self.zone_height] * len(self.flags[1]),
                          placement_fn = self.flags[0] + self.flags[1],
                          rgba = self.team_colors + self.team_colors,
                          make_static = [False] * len(self.flags[0]) + [True] * len(self.flags[1]))
        env.add_module(flags)

        # Add LIDAR visualization to the environment
        if self.lidar > 0 and self.visualize_lidar:
            lidar = LidarSites(n_agents = len(self.agents),
                               n_lidar_per_agent = self.lidar)
            env.add_module(lidar)
        
        return env
    
    def govern_env(self, env: Base):
        # Add floor physics to the environment
        friction = FloorAttributes(friction = self.floor_friction)
        env.add_module(friction)

        # Add gravity to the environment
        gravity = WorldConstants(gravity = self.gravity)
        env.add_module(gravity)
        
        # Split movement by agent
        env = SplitMultiAgentActions(env)

        # Discretize agent movement
        env = DiscretizeActionWrapper(env, 
                                      action_key = 'action_movement')

        # Allow agents to grab boxes, ramps, and flags
        env = GrabObjWrapper(env,
                             body_names = [f'moveable_box{i}' for i in range(np.max(len(self.boxes[0] + self.boxes[1])))] + [f"ramp{i}:ramp" for i in range(len(self.ramps))] + [f"moveable_cylinder{i}" for i in range(len(self.flags[0]))],
                             radius_multiplier = self.grab_radius,
                             obj_in_game_metadata_keys = ['curr_n_boxes', 'curr_n_ramps', 'curr_n_flags'])
        
        # Allow agents to lock boxes
        env = LockObjWrapper(env,
                             body_names = [f'moveable_box{i}' for i in range(np.max(len(self.boxes[0] + self.boxes[1])))],
                             agent_idx_allowed_to_lock = np.arange(len(self.agents[0] + self.agents[1])),
                             lock_type = self.lock_type,
                             radius_multiplier = self.grab_radius,
                             obj_in_game_metadata_keys = ["curr_n_boxes"],
                             agent_allowed_to_lock_keys = None)
        
        # Allow agents to lock ramps
        env = LockObjWrapper(env,
                             body_names = [f'ramp{i}:ramp' for i in range(len(self.ramps))],
                             agent_idx_allowed_to_lock = np.arange(len(self.agents[0] + self.agents[1])),
                             lock_type = self.lock_type, 
                             ac_obs_prefix = 'ramp_',
                             radius_multiplier = self.grab_radius,
                             obj_in_game_metadata_keys = ['curr_n_ramps'],
                             agent_allowed_to_lock_keys = None)
        
        # Observation keys
        keys_self = ['agent_qpos_qvel']
        keys_mask_self = ['mask_aa_obs']
        keys_external = ['agent_qpos_qvel', 'box_obs', 'ramp_obs', 'flag_obs', 'zone_obs']
        keys_mask_external = ['lidar', 'mask_ab_obs', 'mask_ar_obs', 'mask_af_obs', 'mask_ab_obs_spoof']
        keys_copy = ['lidar', 'you_lock', 'team_lock', 'ramp_you_lock', 'ramp_team_lock']

        # Add LIDAR for each agent
        if self.lidar > 0:
            env = Lidar(env,
                        n_lidar_per_agent = self.lidar,
                        visualize_lidar = self.visualize_lidar)
        
        # Add masks for agent-to-agent observations
        env = AgentAgentObsMask2D(env)
        
        if len(self.boxes[0] + self.boxes[1]) > 0:
            # Add masks for agent-to-box observations
            env = AgentGeomObsMask2D(env,
                                     pos_obs_key = 'box_pos',
                                     mask_obs_key = 'mask_ab_obs',
                                     geom_idxs_obs_key = 'box_geom_idxs')
        
        if len(self.ramps) > 0:
            # Add masks for agent-to-ramp observations
            env = AgentGeomObsMask2D(env,
                                     pos_obs_key = 'ramp_pos',
                                     mask_obs_key = 'mask_ar_obs',
                                     geom_idxs_obs_key = 'ramp_geom_idxs')
        
        if len(self.flags[0]) > 0:
            # Add masks for agent-to-flag observations
            env = AgentGeomObsMask2D(env,
                                     pos_obs_key = 'moveable_cylinder_xpos',
                                     mask_obs_key = 'mask_af_obs',
                                     geom_idxs_obs_key = 'moveable_cylinder_geom_idxs')

        # Splits observations for each agent
        env = SplitObservations(env,
                                keys_self = keys_self + keys_mask_self,
                                keys_copy = keys_copy,
                                keys_self_matrices = keys_mask_self)
        
        # Ensure observation dimension matches environment requirements
        env = SpoofEntityWrapper(env, 
                                 total_n_entities = np.max(len(self.boxes[0] + self.boxes[1])),
                                 keys = ['box_obs', 'you_lock', 'team_lock', 'obj_lock'],
                                 mask_keys = ['mask_ab_obs'])
        
        # Apply observation mask to action space
        env = MaskActionWrapper(env,
                                action_key = 'action_pull',
                                mask_keys = ['mask_ab_obs', 'mask_ar_obs', 'mask_af_obs'])

        # Enforce that agents only grab the closest object
        env = GrabClosestWrapper(env)

        # Allow agents to lock all objects
        env = LockAllWrapper(env,
                             remove_object_specific_lock = True)

        # Assign team membership for each agent
        env = TeamMembership(env,
                             team_index = [0] * len(self.agents[0]) + [1] * len(self.agents[1]))
        
        # Apply reward functions for the game
        env = GameRewardWrapper(env,
                                team0 = self.agents[0],
                                team1 = self.agents[1],
                                rew_fn = self.reward_fn,
                                rew_type = self.reward_type)
        
        # Catches Mujoco exceptions
        env = DiscardMujocoExceptionEpisodes(env)

        # Group observations based on key
        env = ConcatenateObsWrapper(env, 
                                    obs_groups = {
                                        'agent_qpos_qvel': ['agent_qpos_qvel'],
                                        'box_obs': ['box_obs', 'you_lock', 'team_lock', 'obj_lock'],
                                        'ramp_obs': ['ramp_obs', 'ramp_you_lock', 'ramp_team_lock', 'ramp_obj_lock'],
                                        'flag_obs': ['moveable_cylinder_obs'],
                                    })

        # Select which keys are passed as the observation space
        env = SelectKeysWrapper(env,
                                keys_self = keys_self,
                                keys_other = keys_external + keys_mask_self + keys_mask_external)

        return env


'''
Designs the environment based on the game scenario.
'''
class GameScenario:
    def __init__(self, playground, mode, floor_size, grid_size, lidar):
        self.playground = playground
        self.mode = mode
        self.playground_constants = {}
        self.playground_constants['floor_size'] = floor_size / 2
        self.playground_constants['grid_size'] = grid_size + 1
        self.playground_constants['lidar'] = lidar
        self.playground_constants['lock_type'] = 'any_lock_specific'
        self.playground_constants['floor_friction'] = 0.2
        self.playground_constants['object_friction'] = 0.01
        self.playground_constants['gravity'] = [0, 0, -50]
        self.playground_constants['door_size'] = int(grid_size / 16)
        self.playground_constants['box_size'] = floor_size / 24
        self.playground_constants['box_length'] = floor_size * (1 / 4 - 2 / grid_size)
        self.playground_constants['box_width'] = 10 / grid_size
        self.playground_constants['box_height'] = 12 / floor_size
        self.playground_constants['flag_diameter'] = floor_size / 75
        self.playground_constants['flag_height'] = 6 / floor_size
        self.playground_constants['zone_diameter'] = floor_size / 50
        self.playground_constants['zone_height'] = 1 / floor_size
        self.playground_constants['grab_radius'] = 6 / floor_size
        self.playground_constants['alignment'] = 1
    
    def get_playground_constants(self):
        return self.playground_constants
    
    @staticmethod
    def coords(grid_size, proportion):
        return int((proportion + 1) / 2 * (grid_size - 1))

    def get_walls(self):
        grid_size = self.playground_constants['grid_size']

        if self.playground == 0:
            self.walls = []
            self.doors = [
                Wall([self.coords(grid_size, 0), self.coords(grid_size, 0)], [self.coords(grid_size, 1), self.coords(grid_size, 0)]),
                Wall([self.coords(grid_size, 0), self.coords(grid_size, 0)], [self.coords(grid_size, 0), self.coords(grid_size, -1)]),
            ]
        elif self.playground == 1:
            self.walls = [
                Wall([self.coords(grid_size, 0), self.coords(grid_size, 0)], [self.coords(grid_size, 0), self.coords(grid_size, -1)]),
                Wall([self.coords(grid_size, 0), self.coords(grid_size, 0)], [self.coords(grid_size, 0), self.coords(grid_size, 1)]),
            ]
            self.doors = [
                Wall([self.coords(grid_size, 0), self.coords(grid_size, 0)], [self.coords(grid_size, -1), self.coords(grid_size, 0)]),
                Wall([self.coords(grid_size, 0), self.coords(grid_size, 0)], [self.coords(grid_size, 1), self.coords(grid_size, 0)]),
            ]
        elif self.playground == 2:
            self.walls = [
                Wall([self.coords(grid_size, 0), self.coords(grid_size, -1)], [self.coords(grid_size, 0), self.coords(grid_size, 1)]),
                Wall([self.coords(grid_size, 1/2), self.coords(grid_size, -1/2)], [self.coords(grid_size, 1), self.coords(grid_size, -1/2)]),
                Wall([self.coords(grid_size, -1/2), self.coords(grid_size, 1/2)], [self.coords(grid_size, -1), self.coords(grid_size, 1/2)]),
            ]
            self.doors = [
                Wall([self.coords(grid_size, 1/2), self.coords(grid_size, -1/2)], [self.coords(grid_size, 1/2), self.coords(grid_size, -1)]),
                Wall([self.coords(grid_size, -1/2), self.coords(grid_size, 1/2)], [self.coords(grid_size, -1/2), self.coords(grid_size, 1)]),
            ]
        
        return (self.walls, self.doors)
    
    def get_agents(self):
        grid_size = self.playground_constants['grid_size']

        if self.playground == 0 or self.playground == 1:
            self.agents0 = [
                partial(object_placement, bounds = ([self.coords(grid_size, 0), self.coords(grid_size, 0)], [self.coords(grid_size, 1), self.coords(grid_size, -1)])),
                partial(object_placement, bounds = ([self.coords(grid_size, 0), self.coords(grid_size, 0)], [self.coords(grid_size, 1), self.coords(grid_size, -1)])),
            ]
            self.agents1 = [
                partial(object_placement, bounds = ([self.coords(grid_size, 0), self.coords(grid_size, 0)], [self.coords(grid_size, -1), self.coords(grid_size, 1)])),
                partial(object_placement, bounds = ([self.coords(grid_size, 0), self.coords(grid_size, 0)], [self.coords(grid_size, -1), self.coords(grid_size, 1)])),
            ]
        elif self.playground == 2:
            self.agents0 = [
                partial(object_placement, bounds = ([self.coords(grid_size, 1/2), self.coords(grid_size, -1/2)], [self.coords(grid_size, 1), self.coords(grid_size, 1/2)])),
                partial(object_placement, bounds = ([self.coords(grid_size, 1/2), self.coords(grid_size, -1/2)], [self.coords(grid_size, 1), self.coords(grid_size, 1/2)])),
            ]
            self.agents1 = [
                partial(object_placement, bounds = ([self.coords(grid_size, -1/2), self.coords(grid_size, 1/2)], [self.coords(grid_size, -1), self.coords(grid_size, -1/2)])),
                partial(object_placement, bounds = ([self.coords(grid_size, -1/2), self.coords(grid_size, 1/2)], [self.coords(grid_size, -1), self.coords(grid_size, -1/2)])),
            ]

        return (self.agents0, self.agents1)
    
    def get_boxes(self):
        grid_size = self.playground_constants['grid_size']

        if self.playground == 0:
            self.boxes = [
                partial(object_placement, bounds = ([self.coords(grid_size, 0), self.coords(grid_size, 0)], [self.coords(grid_size, 1), self.coords(grid_size, -1)])),
                partial(object_placement, bounds = ([self.coords(grid_size, 0), self.coords(grid_size, 0)], [self.coords(grid_size, 1), self.coords(grid_size, -1)])),
            ]
            self.long_boxes = []
        elif self.playground == 1:
            self.boxes = [
                partial(object_placement, bounds = ([self.coords(grid_size, 0), self.coords(grid_size, 0)], [self.coords(grid_size, 1), self.coords(grid_size, -1)])),
                partial(object_placement, bounds = ([self.coords(grid_size, 0), self.coords(grid_size, 0)], [self.coords(grid_size, -1), self.coords(grid_size, 1)])),
            ]
            self.long_boxes = []
        elif self.playground == 2:
            self.boxes = [
                partial(object_placement, bounds = ([self.coords(grid_size, 1/2), self.coords(grid_size, -1/2)], [self.coords(grid_size, 0), self.coords(grid_size, 1/2)])),
                partial(object_placement, bounds = ([self.coords(grid_size, -1/2), self.coords(grid_size, 1/2)], [self.coords(grid_size, 0), self.coords(grid_size, -1/2)])),
            ]
            self.long_boxes = [
                partial(object_placement, bounds = ([self.coords(grid_size, 2 / (grid_size - 1)), self.coords(grid_size, -1/2)],)),
                partial(object_placement, bounds = ([self.coords(grid_size, -1/2 + 2 / (grid_size - 1)), self.coords(grid_size, 1/2)],)),
            ]
        
        return (self.boxes, self.long_boxes)
    
    def get_ramps(self):
        grid_size = self.playground_constants['grid_size']

        if self.playground == 0:
            self.ramps = [
                partial(object_placement, bounds = ([self.coords(grid_size, 0), self.coords(grid_size, 0)], [self.coords(grid_size, 1), self.coords(grid_size, -1)])),
            ]
        elif self.playground == 1:
            self.ramps = [
                partial(object_placement, bounds = ([self.coords(grid_size, 0), self.coords(grid_size, 0)], [self.coords(grid_size, 1), self.coords(grid_size, -1)])),
                partial(object_placement, bounds = ([self.coords(grid_size, 0), self.coords(grid_size, 0)], [self.coords(grid_size, -1), self.coords(grid_size, 1)])),
            ]
        elif self.playground == 2:
            self.ramps = [
                partial(object_placement, bounds = ([self.coords(grid_size, 1/2), self.coords(grid_size, -1/2)], [self.coords(grid_size, 1), self.coords(grid_size, 1/2)])),
                partial(object_placement, bounds = ([self.coords(grid_size, -1/2), self.coords(grid_size, 1/2)], [self.coords(grid_size, -1), self.coords(grid_size, -1/2)])),
            ]
        
        return (self.ramps)
    
    def get_flags(self):
        grid_size = self.playground_constants['grid_size']

        self.flags = [
            partial(object_placement, bounds = ([self.coords(grid_size, -3/4), self.coords(grid_size, 3/4)], [self.coords(grid_size, -1), self.coords(grid_size, 1)])),
            partial(object_placement, bounds = ([self.coords(grid_size, 3/4), self.coords(grid_size, -3/4)], [self.coords(grid_size, 1), self.coords(grid_size, -1)])),
        ]
        if self.mode == 2 or self.mode == 3:
            self.zones = [
                partial(object_placement, bounds = ([self.coords(grid_size, -3/4), self.coords(grid_size, -3/4)], [self.coords(grid_size, -1), self.coords(grid_size, -1)])),
                partial(object_placement, bounds = ([self.coords(grid_size, 3/4), self.coords(grid_size, 3/4)], [self.coords(grid_size, 1), self.coords(grid_size, 1)])),
            ]
        else:
            self.zones = []
        
        return (self.flags, self.zones)
    
    def get_reward(self):
        return None


'''
Keeps track of important statistics that are indicative of game dynamics.
'''
class TrackStatWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def reset(self):
        obs = self.env.reset()
        
        self.initial_statistic = True

        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        # Track statistics at end of episode
        if done:
            self.tracked_statistic = False
            
            info.update({
                'initial_statistic': self.initial_statistic,
                'tracked_statistic': self.tracked_statistic,
            })

        return obs, rew, done, info


'''
Establishes game dynamics (see different reward types below).
Args:
    rew_type (string): can be
        'selfish': agents play selfishly. Seekers recieve 1.0 if they can
            see any hider and -1.0 otherwise. Hiders recieve 1.0 if they are seen by no
            seekers and -1.0 otherwise.
        'joint_mean': agents recieve the mean reward of their team
        'joint_zero_sum': hiders recieve 1.0 only if all hiders are hidden and -1.0 otherwise.
            Seekers recieve 1.0 if any seeker sees a hider.
    reward_scale (float): scales the reward by this factor
'''
class GameRewardWrapper(gym.Wrapper):
    def __init__(self, env, team0, team1, rew_fn=None, rew_type='selfish', reward_scale=1.0):
        super().__init__(env)
        self.n_agents = self.unwrapped.n_agents
        self.rew_fn = rew_fn
        self.rew_type = rew_type
        self.n0 = len(team0)
        self.n1 = len(team1)
        self.reward_scale = reward_scale
        assert self.n0 + self.n1 == self.n_agents, "n0 + n1 must equal n_agents"

        # Agent names are used to plot agent-specific rewards on tensorboard
        self.unwrapped.agent_names = [f'hider{i}' for i in range(self.n0)] + \
                                     [f'seeker{i}' for i in range(self.n1)]

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        if self.rew_fn is None:
            this_rew = np.ones((self.n_agents,))
            this_rew[:self.n0][np.any(obs['mask_aa_obs'][self.n0:, :self.n0], 0)] = -1.0
            this_rew[self.n0:][~np.any(obs['mask_aa_obs'][self.n0:, :self.n0], 1)] = -1.0
        else:
            this_rew = self.rew_fn(self.n0, self.n1)

        if self.rew_type == 'joint_mean':
            this_rew[:self.n0] = this_rew[:self.n0].mean()
            this_rew[self.n0:] = this_rew[self.n0:].mean()
        elif self.rew_type == 'joint_zero_sum':
            this_rew[:self.n0] = np.min(this_rew[:self.n0])
            this_rew[self.n0:] = np.max(this_rew[self.n0:])
        elif self.rew_type == 'selfish':
            pass
        else:
            assert False, f'Hide and Seek reward type {self.rew_type} is not implemented'

        this_rew *= self.reward_scale
        rew += this_rew
        return obs, rew, done, info
    
    def reset(self):
        return self.env.reset()


'''
Places object inside bounds.
'''
def object_placement(grid, obj_size, metadata, random_state, bounds = None):
    if bounds == None:
        return uniform_placement(grid, obj_size, metadata, random_state)

    if len(bounds) == 1:
        return np.array([bounds[0][0], bounds[0][1]])
    
    minX = min(bounds[0][0], bounds[1][0])
    maxX = max(bounds[0][0], bounds[1][0])
    minY = min(bounds[0][1], bounds[1][1])
    maxY = max(bounds[0][1], bounds[1][1])
    return np.array([random_state.randint(minX, maxX),
                    random_state.randint(minY, maxY)])


'''
Makes the environment.
'''
def make_env(playground = 0, mode = 0, floor_size = 12, grid_size = 32, lidar = 0):
    env_generator = GameEnvironment(playground, mode, floor_size, grid_size, lidar)
    env = env_generator.construct_env()
    env.reset()
    env = env_generator.govern_env(env)
    return env
