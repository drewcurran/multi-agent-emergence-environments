import gym
import numpy as np
from copy import deepcopy
from functools import partial
from mae_envs.envs.base import Base
from mae_envs.wrappers.multi_agent import (SplitMultiAgentActions,
                                           SplitObservations, SelectKeysWrapper)
from mae_envs.wrappers.util import (DiscretizeActionWrapper, ConcatenateObsWrapper,
                                    MaskActionWrapper, SpoofEntityWrapper,
                                    DiscardMujocoExceptionEpisodes,
                                    AddConstantObservationsWrapper)
from mae_envs.wrappers.manipulation import (GrabObjWrapper, GrabClosestWrapper,
                                            LockObjWrapper, LockAllWrapper)
from mae_envs.wrappers.lidar import Lidar
from mae_envs.wrappers.line_of_sight import AgentAgentObsMask2D, AgentGeomObsMask2D
from mae_envs.wrappers.prep_phase import PreparationPhase
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
    def __init__(self, lidar = 0, visualize_lidar = False):       
        # World constants
        self.scenario = 'ctf'
        self.floor_size = 10
        self.grid_size = 51
        self.door_size = 2
        self.door_dropout = 0
        self.floor_friction = 0.2
        self.object_friction = 0.01
        self.gravity = [0, 0, -50]
        self.box_size = 0.5
        self.flag_size = 0.2
        self.grab_radius = 0.25 / self.box_size
        self.lock_type = 'any_lock_specific'

        # Team constants
        self.teams = 2
        self.team_players = 2
        self.team_ramps = 2
        self.team_boxes = 1
        self.team_walls = 1
        self.team_flags = 1

        # Game constants
        self.n_players = self.teams * self.team_players
        self.n_boxes = self.teams * self.team_boxes
        self.n_walls = self.teams * self.team_walls
        self.n_ramps = self.teams * self.team_ramps
        self.n_flags = self.teams * self.team_flags

        # Display constants
        self.team_colors = [
            [np.array((66., 235., 244., 255.)) / 255],
            [(1., 0., 0., 1.)]
        ]
        self.horizon = 80
        self.lidar = lidar
        self.visualize_lidar = visualize_lidar

        # Policy constants
        self.substeps = 15
        self.action_lims = (-0.9, 0.9)
        self.deterministic = False
        self.preparation_time = 0.4
        self.reward_type = 'joint_zero_sum'

    def construct_env(self):
        # Create base environment
        env = Base(n_agents = self.n_players,
                   n_substeps = self.substeps,
                   horizon = self.horizon,
                   floor_size = self.floor_size,
                   grid_size = self.grid_size,
                   action_lims = self.action_lims,
                   deterministic_mode = self.deterministic)

        # Add walls to the environment
        walls = WallScenarios(grid_size = self.grid_size,
                              door_size = self.door_size,
                              scenario = self.scenario,
                              walls = [
                                  Wall([coords(self.grid_size, 0), coords(self.grid_size, -1/2)], [coords(self.grid_size, 0), coords(self.grid_size, 1/2)]),
                                  Wall([coords(self.grid_size, -1/3), coords(self.grid_size, -1)], [coords(self.grid_size, -1/3), coords(self.grid_size, -1/2)]),
                                  Wall([coords(self.grid_size, 1/3), coords(self.grid_size, 1)], [coords(self.grid_size, 1/3), coords(self.grid_size, 1/2)]),
                                  Wall([coords(self.grid_size, -9/10), coords(self.grid_size, -1/6)], [coords(self.grid_size, -3/5), coords(self.grid_size, -1/6)]),
                                  Wall([coords(self.grid_size, -9/10), coords(self.grid_size, 1/6)], [coords(self.grid_size, -3/5), coords(self.grid_size, 1/6)]),
                                  Wall([coords(self.grid_size, -9/10), coords(self.grid_size, -1/6)], [coords(self.grid_size, -9/10), coords(self.grid_size, 1/6)]),
                                  Wall([coords(self.grid_size, 9/10), coords(self.grid_size, -1/6)], [coords(self.grid_size, 3/5), coords(self.grid_size, -1/6)]),
                                  Wall([coords(self.grid_size, 9/10), coords(self.grid_size, 1/6)], [coords(self.grid_size, 3/5), coords(self.grid_size, 1/6)]),
                                  Wall([coords(self.grid_size, 9/10), coords(self.grid_size, -1/6)], [coords(self.grid_size, 9/10), coords(self.grid_size, 1/6)]),
                              ],
                              walls_to_split = [
                                  Wall([coords(self.grid_size, -1/3), coords(self.grid_size, -1/2)], [coords(self.grid_size, 1/3), coords(self.grid_size, -1/2)]),
                                  Wall([coords(self.grid_size, -1/3), coords(self.grid_size, 1/2)], [coords(self.grid_size, 1/3), coords(self.grid_size, 1/2)]),
                                  Wall([coords(self.grid_size, -1/3), coords(self.grid_size, 1)], [coords(self.grid_size, -1/3), coords(self.grid_size, 1/2)]),
                                  Wall([coords(self.grid_size, 1/3), coords(self.grid_size, -1)], [coords(self.grid_size, 1/3), coords(self.grid_size, -1/2)]),
                              ],
                              friction = self.object_friction,
                              p_door_dropout = self.door_dropout)
        env.add_module(walls)

        # Add agents to the environment
        agents = Agents(n_agents = self.n_players,
                        placement_fn = [
                            partial(object_placement, bounds = ([coords(self.grid_size, -9/10), coords(self.grid_size, 9/10)], [coords(self.grid_size, -2/3), coords(self.grid_size, 2/3)])),
                            partial(object_placement, bounds = ([coords(self.grid_size, -9/10), coords(self.grid_size, 9/10)], [coords(self.grid_size, -2/3), coords(self.grid_size, 2/3)])),
                            partial(object_placement, bounds = ([coords(self.grid_size, 9/10), coords(self.grid_size, -9/10)], [coords(self.grid_size, 2/3), coords(self.grid_size, -2/3)])),
                            partial(object_placement, bounds = ([coords(self.grid_size, 9/10), coords(self.grid_size, -9/10)], [coords(self.grid_size, 2/3), coords(self.grid_size, -2/3)])),
                        ],
                        color = self.team_colors[0] * self.team_players + self.team_colors[1] * self.team_players,
                        friction = self.object_friction)
        env.add_module(agents)
        env.add_module(AgentManipulation())

        # Add ramps to the environment
        ramps = Ramps(n_ramps = self.n_ramps,
                      placement_fn = [
                            partial(object_placement, bounds = ([coords(self.grid_size, -1/3), coords(self.grid_size, -1/2)], [coords(self.grid_size, 0), coords(self.grid_size, 1/2)])),
                            partial(object_placement, bounds = ([coords(self.grid_size, -1/3), coords(self.grid_size, 9/10)], [coords(self.grid_size, 1/3), coords(self.grid_size, 1/2)])),
                            partial(object_placement, bounds = ([coords(self.grid_size, 1/3), coords(self.grid_size, -1/2)], [coords(self.grid_size, 0), coords(self.grid_size, 1/2)])),
                            partial(object_placement, bounds = ([coords(self.grid_size, 1/3), coords(self.grid_size, -9/10)], [coords(self.grid_size, -1/3), coords(self.grid_size, -1/2)])),
                      ],
                      friction = self.object_friction,
                      pad_ramp_size = True)
        env.add_module(ramps)

        # Add boxes to the environment
        boxes = Boxes(n_boxes = self.n_boxes + self.n_walls,
                      n_elongated_boxes = self.n_walls,
                      placement_fn = [
                            partial(object_placement, bounds = ([coords(self.grid_size, -3/5), coords(self.grid_size, -1/5)], [coords(self.grid_size, -2/5), coords(self.grid_size, 1/5)])),
                            partial(object_placement, bounds = ([coords(self.grid_size, 3/5), coords(self.grid_size, -1/5)], [coords(self.grid_size, 2/5), coords(self.grid_size, 1/5)])),
                            partial(object_placement, bounds = ([coords(self.grid_size, -1/3), coords(self.grid_size, -1/2)], [coords(self.grid_size, 0), coords(self.grid_size, 1/2)])),
                            partial(object_placement, bounds = ([coords(self.grid_size, 1/3), coords(self.grid_size, -1/2)], [coords(self.grid_size, 0), coords(self.grid_size, 1/2)])),
                      ],
                      friction = self.floor_friction,
                      boxid_obs = False,
                      box_only_z_rot = True,
                      alignment = 0)
        env.add_module(boxes)

        # Add flags to the environment
        flags = Cylinders(n_objects = self.n_flags,
                          diameter = self.flag_size,
                          height = self.flag_size * 2,
                          placement_fn = [
                              partial(object_placement, bounds = ([coords(self.grid_size, -3/4), coords(self.grid_size, 0)],)),
                              partial(object_placement, bounds = ([coords(self.grid_size, 3/4), coords(self.grid_size, 0)],)),
                          ],
                          rgba = self.team_colors,
                          make_static = False)
        env.add_module(flags)

        # Add zones to the environment
        zones = Cylinders(n_objects = self.n_flags,
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
            lidar = LidarSites(n_agents = self.n_players,
                               n_lidar_per_agent = self.lidar)
            env.add_module(lidar)
        
        return env
    
    def govern_env(self, env):
        # Self constants
        keys_self = ['agent_qpos_qvel', 'hider', 'prep_obs']

        # Masked self constants
        keys_mask_self = ['mask_aa_obs']

        # External constants
        keys_external = ['agent_qpos_qvel', 'mask_ab_obs', 'box_obs', 'mask_af_obs', 'flag_obs', 'ramp_obs']

        # Masked external constants
        keys_mask_external = ['mask_ab_obs', 'mask_af_obs', 'mask_ar_obs', 'lidar', 'mask_ab_obs_spoof', 'mask_af_obs_spoof']

        # Copy constants
        keys_copy = ['you_lock', 'team_lock', 'ramp_you_lock', 'ramp_team_lock', 'lidar']

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
                             team_index = np.append(np.zeros((self.team_players,)), np.ones((self.team_players,))))
        
        # Add masks for agent-to-agent observations
        env = AgentAgentObsMask2D(env)

        # Apply reward functions for the game
        env = GameRewardWrapper(env,
                                n_hiders = self.team_players,
                                n_seekers = self.team_players,
                                rew_type = self.reward_type)
        
        # TODO: No need for prep phase
        env = PreparationPhase(env,
                               prep_fraction = self.preparation_time)
        
        # Discretize agent actions
        env = DiscretizeActionWrapper(env, 
                                      action_key = 'action_movement')
        
        # Add masks for agent-to-box observations
        env = AgentGeomObsMask2D(env,
                                 pos_obs_key = 'box_pos',
                                 mask_obs_key = 'mask_ab_obs',
                                 geom_idxs_obs_key = 'box_geom_idxs')
        
        # TODO: Constant observations given to hiders
        hider_obs = np.array([[1]] * self.team_players + [[0]] * self.team_players)
        env = AddConstantObservationsWrapper(env,
                                             new_obs = {'hider': hider_obs})
        
        # Add masks for agent-to-ramp observations
        env = AgentGeomObsMask2D(env,
                                 pos_obs_key = 'ramp_pos',
                                 mask_obs_key = 'mask_ar_obs',
                                 geom_idxs_obs_key = 'ramp_geom_idxs')

        # Add ability for agents to grab boxes, ramps, and cylinders
        # TODO: cylinders
        env = GrabObjWrapper(env,
                             body_names = [f'moveable_box{i}' for i in range(np.max(self.n_boxes + self.n_walls))] + ([f"ramp{i}:ramp" for i in range(self.n_ramps)]),
                             radius_multiplier = self.grab_radius,
                             obj_in_game_metadata_keys = ['curr_n_boxes', 'curr_n_ramps'])
        
        # Add ability for agents to lock boxes
        env = LockObjWrapper(env,
                             body_names = [f'moveable_box{i}' for i in range(np.max(self.n_boxes + self.n_walls))],
                             agent_idx_allowed_to_lock = np.arange(self.n_players),
                             lock_type = self.lock_type,
                             radius_multiplier = self.grab_radius,
                             obj_in_game_metadata_keys = ["curr_n_boxes"],
                             agent_allowed_to_lock_keys = None)
        
        # Add ability for agents to lock ramps
        env = LockObjWrapper(env,
                             body_names = [f'ramp{i}:ramp' for i in range(self.n_ramps)],
                             agent_idx_allowed_to_lock = np.arange(self.n_players),
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
                                 total_n_entities = np.max(self.n_boxes + self.n_walls),
                                 keys = ['box_obs', 'you_lock', 'team_lock', 'obj_lock'],
                                 mask_keys = ['mask_ab_obs'])
        
        # Gives agents the ability to lock all objects
        env = LockAllWrapper(env,
                             remove_object_specific_lock = True)
        
        # Adds masks for possible actions
        env = MaskActionWrapper(env,
                                action_key = 'action_pull',
                                mask_keys = ['mask_ab_obs', 'mask_ar_obs'])

        # Enforce that agents only grab the closest object
        env = GrabClosestWrapper(env)
        
        # Catches Mujoco exceptions
        env = DiscardMujocoExceptionEpisodes(env)

        # Groups observations based on key
        env = ConcatenateObsWrapper(env, 
                                    obs_groups = {'agent_qpos_qvel': ['agent_qpos_qvel', 'hider', 'prep_obs'],
                                                  'box_obs': ['box_obs', 'you_lock', 'team_lock', 'obj_lock'],
                                                  'ramp_obs': ['ramp_obs', 'ramp_you_lock', 'ramp_team_lock', 'ramp_obj_lock']})
        
        # Selects keys for final observations
        env = SelectKeysWrapper(env,
                                keys_self = keys_self,
                                keys_other = keys_external + keys_mask_self + keys_mask_external)

        return env



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
    def __init__(self, env, n_hiders, n_seekers, rew_type='selfish', reward_scale=1.0):
        super().__init__(env)
        self.n_agents = self.unwrapped.n_agents
        self.rew_type = rew_type
        self.n_hiders = n_hiders
        self.n_seekers = n_seekers
        self.reward_scale = reward_scale
        assert n_hiders + n_seekers == self.n_agents, "n_hiders + n_seekers must equal n_agents"

        self.metadata['n_hiders'] = n_hiders
        self.metadata['n_seekers'] = n_seekers

        # Agent names are used to plot agent-specific rewards on tensorboard
        self.unwrapped.agent_names = [f'hider{i}' for i in range(self.n_hiders)] + \
                                     [f'seeker{i}' for i in range(self.n_seekers)]

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        this_rew = np.ones((self.n_agents,))
        this_rew[:self.n_hiders][np.any(obs['mask_aa_obs'][self.n_hiders:, :self.n_hiders], 0)] = -1.0
        this_rew[self.n_hiders:][~np.any(obs['mask_aa_obs'][self.n_hiders:, :self.n_hiders], 1)] = -1.0

        if self.rew_type == 'joint_mean':
            this_rew[:self.n_hiders] = this_rew[:self.n_hiders].mean()
            this_rew[self.n_hiders:] = this_rew[self.n_hiders:].mean()
        elif self.rew_type == 'joint_zero_sum':
            this_rew[:self.n_hiders] = np.min(this_rew[:self.n_hiders])
            this_rew[self.n_hiders:] = np.max(this_rew[self.n_hiders:])
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
        self.n_hiders = self.metadata['n_hiders']

    def reset(self):
        self.prev_obs = self.env.reset()
        self.this_team = self.metadata['team_index'] == self.team_idx

        return deepcopy(self.prev_obs)

    def step(self, action):
        is_caught = np.any(self.prev_obs['mask_aa_obs'][self.n_hiders:, :self.n_hiders])
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
def make_env():
    env_generator = GameEnvironment(lidar = 5, visualize_lidar = True)
    env = env_generator.construct_env()
    env.reset()
    env = env_generator.govern_env(env)
    return env
