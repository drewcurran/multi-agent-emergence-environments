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
    def __init__(self, playground, mode):
        # World constants
        self.floor_friction = 0.2
        self.object_friction = 0.01
        self.gravity = [0, 0, -50]
        self.lock_type = 'any_lock_specific'

        # Playground constants
        self.scenario = GameScenario(playground, mode)
        self.playground_constants = self.scenario.get_playground_constants()
        for key, value in self.playground_constants.items(): 
            setattr(self, key, value)
        self.walls = self.scenario.get_walls()
        self.agents = self.scenario.get_agents()
        self.boxes = self.scenario.get_boxes()
        self.ramps = self.scenario.get_ramps()

        # Display constants
        self.team_colors = [
            [np.array((66., 235., 244., 255.)) / 255],
            [(1., 0., 0., 1.)]
        ]
        self.horizon = 80
        self.lidar = 0
        self.visualize_lidar = False

        # Policy constants
        self.substeps = 15
        self.action_lims = (-0.9, 0.9)
        self.deterministic = False
        self.preparation_time = 0.4
        self.reward_type = 'joint_zero_sum'

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
                      placement_fn = self.boxes[0] + self.boxes[1],
                      friction = self.floor_friction,
                      boxid_obs = False,
                      box_only_z_rot = True,
                      alignment = 0)
        env.add_module(boxes)

        # Add ramps to the environment
        ramps = Ramps(n_ramps = len(self.ramps),
                      placement_fn = self.ramps,
                      friction = self.object_friction,
                      pad_ramp_size = True)
        env.add_module(ramps)

        # Add flags to the environment
        flags = Cylinders(n_objects = 2,
                          diameter = self.flag_size,
                          height = self.flag_size * 2,
                          placement_fn = [
                              partial(object_placement, bounds = ([coords(self.grid_size, -3/4), coords(self.grid_size, 3/4)],)),
                              partial(object_placement, bounds = ([coords(self.grid_size, 3/4), coords(self.grid_size, -3/4)],)),
                          ],
                          rgba = self.team_colors,
                          make_static = False)
        env.add_module(flags)

        # Add zones to the environment
        zones = Cylinders(n_objects = 2,
                          diameter = self.flag_size * 2,
                          height = self.flag_size / 4,
                          placement_fn = [
                              partial(object_placement, bounds = ([coords(self.grid_size, 3/4), coords(self.grid_size, 3/4)],)),
                              partial(object_placement, bounds = ([coords(self.grid_size, -3/4), coords(self.grid_size, -3/4)],)),
                          ],
                          rgba = self.team_colors,
                          make_static = True)
        env.add_module(zones)

        # Add LIDAR visualization to the environment
        if self.lidar > 0 and self.visualize_lidar:
            lidar = LidarSites(n_agents = len(self.agents),
                               n_lidar_per_agent = self.lidar)
            env.add_module(lidar)
        
        return env
    
    def govern_env(self, env: Base):
        # Self constants
        keys_self = ['agent_qpos_qvel']

        # External constants
        keys_external = ['agent_qpos_qvel', 'box_obs', 'ramp_obs', 'flag_obs', 'zone_obs']

        # Masked self constants
        keys_mask_self = ['mask_aa_obs']

        # Masked external constants
        keys_mask_external = ['lidar', 'mask_ab_obs', 'mask_ar_obs', 'mask_af_obs', 'mask_az_obs', 'mask_ab_obs_spoof']

        # Agent independent constants
        keys_copy = ['lidar', 'you_lock', 'team_lock', 'ramp_you_lock', 'ramp_team_lock']

        # Add floor physics to the environment
        friction = FloorAttributes(friction = self.floor_friction)
        env.add_module(friction)

        # Add gravity to the environment
        gravity = WorldConstants(gravity = self.gravity)
        env.add_module(gravity)

        # Creates a dictionary for agent actions
        env = SplitMultiAgentActions(env)
        
        # Assign team membership for each agent
        env = TeamMembership(env, 
                             team_index = np.append(np.zeros((len(self.agents[0]),)), np.ones((len(self.agents[1]),))))
        
        # Add masks for agent-to-agent observations
        env = AgentAgentObsMask2D(env)

        # Apply reward functions for the game
        env = GameRewardWrapper(env,
                                n0 = len(self.agents[0]),
                                n1 = len(self.agents[1]),
                                rew_type = self.reward_type)
        
        # Discretize agent actions
        env = DiscretizeActionWrapper(env, 
                                      action_key = 'action_movement')
        
        # Add masks for agent-to-box observations
        env = AgentGeomObsMask2D(env,
                                 pos_obs_key = 'box_pos',
                                 mask_obs_key = 'mask_ab_obs',
                                 geom_idxs_obs_key = 'box_geom_idxs')
        
        # Add masks for agent-to-ramp observations
        env = AgentGeomObsMask2D(env,
                                 pos_obs_key = 'ramp_pos',
                                 mask_obs_key = 'mask_ar_obs',
                                 geom_idxs_obs_key = 'ramp_geom_idxs')
        
        # Add masks for agent-to-flag observations
        env = AgentGeomObsMask2D(env,
                                 pos_obs_key = 'moveable_cylinder_xpos',
                                 mask_obs_key = 'mask_af_obs',
                                 geom_idxs_obs_key = 'moveable_cylinder_geom_idxs')

        # Add ability for agents to grab boxes, ramps, and flags
        env = GrabObjWrapper(env,
                             body_names = [f'moveable_box{i}' for i in range(np.max(len(self.boxes[0] + self.boxes[1])))] + [f"ramp{i}:ramp" for i in range(len(self.ramps))] + [f"moveable_cylinder{i}" for i in range(2)],
                             radius_multiplier = self.grab_radius,
                             obj_in_game_metadata_keys = ['curr_n_boxes', 'curr_n_ramps', 'curr_n_flags'])
        
        # Add ability for agents to lock boxes
        env = LockObjWrapper(env,
                             body_names = [f'moveable_box{i}' for i in range(np.max(len(self.boxes[0] + self.boxes[1])))],
                             agent_idx_allowed_to_lock = np.arange(len(self.agents[0] + self.agents[1])),
                             lock_type = self.lock_type,
                             radius_multiplier = self.grab_radius,
                             obj_in_game_metadata_keys = ["curr_n_boxes"],
                             agent_allowed_to_lock_keys = None)
        
        # Add ability for agents to lock ramps
        env = LockObjWrapper(env,
                             body_names = [f'ramp{i}:ramp' for i in range(len(self.ramps))],
                             agent_idx_allowed_to_lock = np.arange(len(self.agents[0] + self.agents[1])),
                             lock_type = self.lock_type, 
                             ac_obs_prefix = 'ramp_',
                             radius_multiplier = self.grab_radius,
                             obj_in_game_metadata_keys = ['curr_n_ramps'],
                             agent_allowed_to_lock_keys = None)
        
        # Adds LIDAR for each agent
        if self.lidar > 0:
            env = Lidar(env,
                        n_lidar_per_agent = self.lidar,
                        visualize_lidar = self.visualize_lidar)

        # Splits observations for each agent
        env = SplitObservations(env,
                                keys_self = keys_self + keys_mask_self,
                                keys_copy = keys_copy,
                                keys_self_matrices = keys_mask_self)
        
        # Adds extra entities to ensure environment matches
        env = SpoofEntityWrapper(env, 
                                 total_n_entities = np.max(len(self.boxes[0] + self.boxes[1])),
                                 keys = ['box_obs', 'you_lock', 'team_lock', 'obj_lock'],
                                 mask_keys = ['mask_ab_obs'])
        
        # Gives agents the ability to lock all objects
        env = LockAllWrapper(env,
                             remove_object_specific_lock = True)
        
        # Adds masks for possible actions
        env = MaskActionWrapper(env,
                                action_key = 'action_pull',
                                mask_keys = ['mask_ab_obs', 'mask_ar_obs', 'mask_af_obs'])

        # Enforce that agents only grab the closest object
        env = GrabClosestWrapper(env)
        
        # Catches Mujoco exceptions
        env = DiscardMujocoExceptionEpisodes(env)

        # Groups observations based on key
        env = ConcatenateObsWrapper(env, 
                                    obs_groups = {'agent_qpos_qvel': ['agent_qpos_qvel'],
                                                  'box_obs': ['box_obs', 'you_lock', 'team_lock', 'obj_lock'],
                                                  'ramp_obs': ['ramp_obs', 'ramp_you_lock', 'ramp_team_lock', 'ramp_obj_lock'],
                                                  'flag_obs': ['moveable_cylinder_obs'],
                                                  })
        
        # Selects keys for final observations
        env = SelectKeysWrapper(env,
                                keys_self = keys_self,
                                keys_other = keys_external + keys_mask_self + keys_mask_external)

        return env


'''
Designs the environment based on the phase.
'''
class GameScenario:
    def __init__(self, playground, mode):   # TODO: Make function of grid_size
        self.playground = playground
        self.mode = mode
        self.playground_constants = {}
        self.get_playground_constants()
        self.walls = []
        self.doors = []
        self.agents0 = []
        self.agents1 = []
        self.boxes = []
        self.long_boxes = []
        self.ramps = []
    
    def get_playground_constants(self):
        self.playground_constants['floor_size'] = 6
        self.playground_constants['grid_size'] = 31
        self.playground_constants['door_size'] = 2
        self.playground_constants['box_size'] = 0.5
        self.playground_constants['flag_size'] = 0.2
        self.playground_constants['grab_radius'] = 0.25 / self.playground_constants['box_size']

        return self.playground_constants

    def get_walls(self):
        grid_size = self.playground_constants['grid_size']

        if self.playground == 0:
            self.walls = []
            self.doors = [
                Wall([coords(grid_size, 0), coords(grid_size, 0)], [coords(grid_size, 1), coords(grid_size, 0)]),
                Wall([coords(grid_size, 0), coords(grid_size, 0)], [coords(grid_size, 0), coords(grid_size, -1)]),
            ]
        elif self.playground == 1:
            self.walls = [
                Wall([coords(grid_size, 0), coords(grid_size, 0)], [coords(grid_size, 0), coords(grid_size, -1)]),
                Wall([coords(grid_size, 0), coords(grid_size, 0)], [coords(grid_size, 0), coords(grid_size, 1)]),
            ]
            self.doors = [
                Wall([coords(grid_size, 0), coords(grid_size, 0)], [coords(grid_size, -1), coords(grid_size, 0)]),
                Wall([coords(grid_size, 0), coords(grid_size, 0)], [coords(grid_size, 1), coords(grid_size, 0)]),
            ]
        elif self.playground == 2:
            self.walls = [
                Wall([coords(grid_size, 0), coords(grid_size, 0)], [coords(grid_size, 0), coords(grid_size, -1)]),
                Wall([coords(grid_size, 0), coords(grid_size, 0)], [coords(grid_size, 0), coords(grid_size, 1)]),
            ]
            self.doors = [
                Wall([coords(grid_size, 0), coords(grid_size, 0)], [coords(grid_size, -1), coords(grid_size, 0)]),
                Wall([coords(grid_size, 0), coords(grid_size, 0)], [coords(grid_size, 1), coords(grid_size, 0)]),
            ]
        
        return (self.walls, self.doors)
    
    def get_agents(self):
        grid_size = self.playground_constants['grid_size']

        if self.playground == 0 or self.playground == 1:
            self.agents0 = [
                partial(object_placement, bounds = ([coords(grid_size, 0), coords(grid_size, 0)], [coords(grid_size, 1), coords(grid_size, -1)])),
                partial(object_placement, bounds = ([coords(grid_size, 0), coords(grid_size, 0)], [coords(grid_size, 1), coords(grid_size, -1)])),
            ]
            self.agents1 = [
                partial(object_placement, bounds = ([coords(grid_size, 0), coords(grid_size, 0)], [coords(grid_size, -1), coords(grid_size, 1)])),
                partial(object_placement, bounds = ([coords(grid_size, 0), coords(grid_size, 0)], [coords(grid_size, -1), coords(grid_size, 1)])),
            ]
        elif self.playground == 2:
            self.agents0 = [
                partial(object_placement, bounds = ([coords(grid_size, 0), coords(grid_size, 0)], [coords(grid_size, 1), coords(grid_size, -1)])),
                partial(object_placement, bounds = ([coords(grid_size, 0), coords(grid_size, 0)], [coords(grid_size, 1), coords(grid_size, -1)])),
            ]
            self.agents1 = [
                partial(object_placement, bounds = ([coords(grid_size, 0), coords(grid_size, 0)], [coords(grid_size, -1), coords(grid_size, 1)])),
                partial(object_placement, bounds = ([coords(grid_size, 0), coords(grid_size, 0)], [coords(grid_size, -1), coords(grid_size, 1)])),
            ]

        return (self.agents0, self.agents1)
    
    def get_boxes(self):
        grid_size = self.playground_constants['grid_size']

        if self.playground == 0:
            self.boxes = [
                partial(object_placement, bounds = ([coords(grid_size, 0), coords(grid_size, 0)], [coords(grid_size, 1), coords(grid_size, -1)])),
                partial(object_placement, bounds = ([coords(grid_size, 0), coords(grid_size, 0)], [coords(grid_size, 1), coords(grid_size, -1)])),
            ]
            self.long_boxes = []
        elif self.playground == 1:
            self.boxes = [
                partial(object_placement, bounds = ([coords(grid_size, 0), coords(grid_size, 0)], [coords(grid_size, 1), coords(grid_size, -1)])),
                partial(object_placement, bounds = ([coords(grid_size, 0), coords(grid_size, 0)], [coords(grid_size, -1), coords(grid_size, 1)])),
            ]
            self.long_boxes = []
        elif self.playground == 2:
            self.boxes = [
                partial(object_placement, bounds = ([coords(grid_size, 0), coords(grid_size, 0)], [coords(grid_size, 1), coords(grid_size, -1)])),
                partial(object_placement, bounds = ([coords(grid_size, 0), coords(grid_size, 0)], [coords(grid_size, -1), coords(grid_size, 1)])),
            ]
            self.long_boxes = []
        
        return (self.boxes, self.long_boxes)
    
    def get_ramps(self):
        grid_size = self.playground_constants['grid_size']

        if self.playground == 0:
            self.ramps = [
                partial(object_placement, bounds = ([coords(grid_size, 0), coords(grid_size, 0)], [coords(grid_size, 1), coords(grid_size, -1)])),
            ]
        elif self.playground == 1:
            self.ramps = [
                partial(object_placement, bounds = ([coords(grid_size, 0), coords(grid_size, 0)], [coords(grid_size, 1), coords(grid_size, -1)])),
                partial(object_placement, bounds = ([coords(grid_size, 0), coords(grid_size, 0)], [coords(grid_size, -1), coords(grid_size, 1)])),
            ]
        elif self.playground == 2:
            self.ramps = [
                partial(object_placement, bounds = ([coords(grid_size, 0), coords(grid_size, 0)], [coords(grid_size, 1), coords(grid_size, -1)])),
                partial(object_placement, bounds = ([coords(grid_size, 0), coords(grid_size, 0)], [coords(grid_size, -1), coords(grid_size, 1)])),
            ]
        
        return (self.ramps)


'''
Keeps track of important statistics that are indicative of game dynamics.
'''
class TrackStatWrapper(gym.Wrapper):
    def __init__(self, env, n_boxes, n_ramps, n_food):
        super().__init__(env)
        self.n_boxes = n_boxes
        self.n_ramps = n_ramps
        self.n_food = n_food

    def reset(self):
        obs = self.env.reset()
        if self.n_boxes > 0:
            self.box_pos_start = obs['box_pos']
        if self.n_ramps > 0:
            self.ramp_pos_start = obs['ramp_pos']
        if self.n_food > 0:
            self.total_food_eaten = np.sum(obs['food_eat'])

        self.in_prep_phase = True

        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        if self.n_food > 0:
            self.total_food_eaten += np.sum(obs['food_eat'])

        if self.in_prep_phase and obs['prep_obs'][0, 0] == 1.0:
            # Track statistics at end of preparation phase
            self.in_prep_phase = False

            if self.n_boxes > 0:
                self.max_box_move_prep = np.max(np.linalg.norm(obs['box_pos'] - self.box_pos_start, axis=-1))
                self.num_box_lock_prep = np.sum(obs['obj_lock'])
            if self.n_ramps > 0:
                self.max_ramp_move_prep = np.max(np.linalg.norm(obs['ramp_pos'] - self.ramp_pos_start, axis=-1))
                if 'ramp_obj_lock' in obs:
                    self.num_ramp_lock_prep = np.sum(obs['ramp_obj_lock'])
            if self.n_food > 0:
                self.total_food_eaten_prep = self.total_food_eaten

        if done:
            # Track statistics at end of episode
            if self.n_boxes > 0:
                self.max_box_move = np.max(np.linalg.norm(obs['box_pos'] - self.box_pos_start, axis=-1))
                self.num_box_lock = np.sum(obs['obj_lock'])
                info.update({
                    'max_box_move_prep': self.max_box_move_prep,
                    'max_box_move': self.max_box_move,
                    'num_box_lock_prep': self.num_box_lock_prep,
                    'num_box_lock': self.num_box_lock})

            if self.n_ramps > 0:
                self.max_ramp_move = np.max(np.linalg.norm(obs['ramp_pos'] - self.ramp_pos_start, axis=-1))
                info.update({
                    'max_ramp_move_prep': self.max_ramp_move_prep,
                    'max_ramp_move': self.max_ramp_move})
                if 'ramp_obj_lock' in obs:
                    self.num_ramp_lock = np.sum(obs['ramp_obj_lock'])
                    info.update({
                        'num_ramp_lock_prep': self.num_ramp_lock_prep,
                        'num_ramp_lock': self.num_ramp_lock})

            if self.n_food > 0:
                info.update({
                    'food_eaten': self.total_food_eaten,
                    'food_eaten_prep': self.total_food_eaten_prep})

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
    def __init__(self, env, n0, n1, rew_type='selfish', reward_scale=1.0):
        super().__init__(env)
        self.n_agents = self.unwrapped.n_agents
        self.rew_type = rew_type
        self.n0 = n0
        self.n1 = n1
        self.reward_scale = reward_scale
        assert n0 + n1 == self.n_agents, "n0 + n1 must equal n_agents"

        self.metadata['n_hiders'] = n0
        self.metadata['n_seekers'] = n1
        self.metadata['hiders_score'] = 0
        self.metadata['seekers_score'] = 0
        self.metadata['prev_score_diff'] = 0

        # Agent names are used to plot agent-specific rewards on tensorboard
        self.unwrapped.agent_names = [f'hider{i}' for i in range(self.n0)] + \
                                     [f'seeker{i}' for i in range(self.n1)]

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        difference = self.metadata['hiders_score'] - self.metadata['seekers_score']
        this_rew = np.ones((self.n_agents,))
        this_rew[:self.n0] = difference - self.metadata['prev_score_diff'] / 2
        this_rew[self.n0:] = self.metadata['prev_score_diff'] / 2 - difference
        self.metadata['prev_score_diff'] = difference

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
Masks a (binary) action with some probability if agent or any of its teammates was being observed
by opponents at any of the last n_latency time step.

Args:
    team_idx (int): Team index (e.g. 0 = hiders) of team whose actions are
                    masked
    action_key (string): key of action to be masked
'''
class MaskUnseenAction(gym.Wrapper):
    def __init__(self, env, team_idx, action_key):
        super().__init__(env)
        self.team_idx = team_idx
        self.action_key = action_key
        self.n_agents = self.unwrapped.n_agents
        self.n0 = self.metadata['n_hiders']

    def reset(self):
        self.prev_obs = self.env.reset()
        self.this_team = self.metadata['team_index'] == self.team_idx

        return deepcopy(self.prev_obs)

    def step(self, action):
        is_caught = np.any(self.prev_obs['mask_aa_obs'][self.n0:, :self.n0])
        if is_caught:
            action[self.action_key][self.this_team] = 0

        self.prev_obs, rew, done, info = self.env.step(action)
        return deepcopy(self.prev_obs), rew, done, info


'''
Places object inside bounds.
'''
def coords(grid_size, proportion):
    return int((proportion + 1) / 2 * grid_size - 0.5)


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
def make_env(playground = 1, mode = 0):
    env_generator = GameEnvironment(playground, mode)
    env = env_generator.construct_env()
    env.reset()
    env = env_generator.govern_env(env)
    return env
