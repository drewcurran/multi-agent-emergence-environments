import gym
import numpy as np
from copy import deepcopy
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
from mae_envs.wrappers.line_of_sight import (AgentAgentObsMask2D, AgentGeomObsMask2D,
                                             AgentSiteObsMask2D)
from mae_envs.wrappers.prep_phase import (PreparationPhase, NoActionsInPrepPhase,
                                          MaskPrepPhaseAction)
from mae_envs.wrappers.limit_mvmnt import RestrictAgentsRect
from mae_envs.wrappers.team import TeamMembership
from mae_envs.wrappers.food import FoodHealthWrapper, AlwaysEatWrapper
from mae_envs.modules.agents import Agents, AgentManipulation
from mae_envs.modules.walls import WallScenarios
from mae_envs.modules.objects import Boxes, Ramps, LidarSites
from mae_envs.modules.food import Food
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
        self.grid_size = 50
        self.door_size = 2
        self.door_dropout = 0
        self.floor_friction = 0.2
        self.object_friction = 0.01
        self.gravity = [0, 0, -50]
        self.flag_size = 0.1

        # Team constants
        self.teams = 2
        self.team_players = 2
        self.team_ramps = 1
        self.team_boxes = 1
        self.team_walls = 1
        self.team_flags = 1

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

    def player_placement(self):
        pass

    def ramp_placement(self):
        pass

    def box_placement(self):
        pass

    def wall_placement(self):
        pass
    
    def make_env(self,
             box_size=0.5, boxid_obs=False, box_only_z_rot=True,
             rew_type='joint_zero_sum', lock_ramp=True,
             lock_type='any_lock_specific',
             lock_grab_radius=0.25, grab_exclusive=False
             ):
        n_players = self.teams * self.team_players
        n_boxes = self.teams * self.team_boxes
        n_ramps = self.teams * self.team_ramps
        n_flags = self.teams * self.team_flags

        keys_self = ['agent_qpos_qvel', 'hider', 'prep_obs']
        keys_mask_self = ['mask_aa_obs']
        keys_external = ['agent_qpos_qvel', 'mask_ab_obs', 'box_obs', 'mask_af_obs', 'food_obs', 'ramp_obs']
        keys_copy = ['you_lock', 'team_lock', 'ramp_you_lock', 'ramp_team_lock', 'lidar']
        keys_mask_external = ['mask_ab_obs', 'mask_af_obs', 'mask_ar_obs', 'lidar']

        env = Base(n_agents = n_players,
                   n_substeps = self.substeps,
                   horizon = self.horizon,
                   floor_size = self.floor_size,
                   grid_size = self.grid_size,
                   action_lims = self.action_lims,
                   deterministic_mode = self.deterministic)
        
        walls = WallScenarios(grid_size = self.grid_size,
                              door_size = self.door_size,
                              scenario = self.scenario,
                              friction = self.object_friction,
                              p_door_dropout = self.door_dropout)
        env.add_module(walls)

        agent_placement_fn = uniform_placement
        agents = Agents(n_agents = n_players,
                        placement_fn = agent_placement_fn,
                        color = self.team_colors[0] * self.team_players + self.team_colors[1] * self.team_players,
                        friction = self.object_friction)
        env.add_module(agents)
        env.add_module(AgentManipulation())

        ramp_placement_fn = uniform_placement
        ramps = Ramps(n_ramps = n_ramps,
                      placement_fn = ramp_placement_fn,
                      friction = self.object_friction,
                      pad_ramp_size = True)
        env.add_module(ramps)

        box_placement_fn = uniform_placement
        boxes = Boxes(n_boxes = self.teams * (self.team_boxes + self.team_walls),
                      n_elongated_boxes = self.teams * self.team_walls,
                      placement_fn = box_placement_fn,
                      friction = self.floor_friction,
                      boxid_obs = boxid_obs,
                      box_only_z_rot = box_only_z_rot)
        env.add_module(boxes)

        food_placement = uniform_placement
        flags = Food(n_food = n_flags,
                     food_size = self.flag_size,
                     placement_fn = food_placement)
        env.add_module(flags)

        friction = FloorAttributes(friction = self.floor_friction)
        env.add_module(friction)

        gravity = WorldConstants(gravity = self.gravity)
        env.add_module(gravity)

        if self.lidar > 0 and self.visualize_lidar:
            lidar = LidarSites(n_agents = n_players,
                               n_lidar_per_agent = self.lidar)
            env.add_module(lidar)


        env.reset()

        env = SplitMultiAgentActions(env)

        env = TeamMembership(env, 
                             team_index = np.append(np.zeros((self.team_players,)), np.ones((self.team_players,))))
        
        env = AgentAgentObsMask2D(env)

        env = GameRewardWrapper(env,
                                n_hiders = self.team_players,
                                n_seekers = self.team_players,
                                rew_type = rew_type)
        
        env = PreparationPhase(env,
                               prep_fraction = self.preparation_time)
        
        env = DiscretizeActionWrapper(env, 
                                      action_key = 'action_movement')
        
        env = AgentGeomObsMask2D(env,
                                 pos_obs_key = 'box_pos',
                                 mask_obs_key = 'mask_ab_obs',
                                 geom_idxs_obs_key = 'box_geom_idxs')
        
        # TODO: Constant observations given to hiders
        hider_obs = np.array([[1]] * self.team_players + [[0]] * self.team_players)
        env = AddConstantObservationsWrapper(env, new_obs={'hider': hider_obs})

        # TODO: Food used as health
        env = AgentSiteObsMask2D(env,
                                 pos_obs_key = 'food_pos',
                                 mask_obs_key = 'mask_af_obs')
        env = FoodHealthWrapper(env,
                                respawn_time = np.inf,
                                eat_thresh = self.flag_size,
                                max_food_health = 1,
                                food_rew_type = 'selfish',
                                reward_scale = 1.0)
        env = MaskActionWrapper(env, 'action_eat_food', ['mask_af_obs'])
        env = MaskUnseenAction(env, 0, 'action_eat_food')
        env = AlwaysEatWrapper(env,
                               agent_idx_allowed = np.arange(n_players))
        
        env = AgentGeomObsMask2D(env,
                                 pos_obs_key = 'ramp_pos',
                                 mask_obs_key = 'mask_ar_obs',
                                 geom_idxs_obs_key = 'ramp_geom_idxs')

        env = GrabObjWrapper(env,
                             body_names = [f'moveable_box{i}' for i in range(np.max(n_boxes))] + ([f"ramp{i}:ramp" for i in range(n_ramps)]),
                             radius_multiplier = lock_grab_radius / box_size,
                             grab_exclusive = grab_exclusive,
                             obj_in_game_metadata_keys = ['curr_n_boxes', 'curr_n_ramps'])
        
        env = LockObjWrapper(env,
                             body_names = [f'moveable_box{i}' for i in range(np.max(n_boxes))],
                             agent_idx_allowed_to_lock = np.arange(n_players),
                             lock_type = lock_type,
                             radius_multiplier = lock_grab_radius / box_size,
                             obj_in_game_metadata_keys = ["curr_n_boxes"],
                             agent_allowed_to_lock_keys = None)
        
        env = LockObjWrapper(env, body_names=[f'ramp{i}:ramp' for i in range(n_ramps)],
                                 agent_idx_allowed_to_lock=np.arange(n_players),
                                 lock_type=lock_type, ac_obs_prefix='ramp_',
                                 radius_multiplier=lock_grab_radius / box_size,
                                 obj_in_game_metadata_keys=['curr_n_ramps'],
                                 agent_allowed_to_lock_keys=None)
        
        if self.lidar > 0:
            env = Lidar(env,
                    n_lidar_per_agent = self.lidar,
                    visualize_lidar = self.visualize_lidar)

        env = SplitObservations(env,
                                keys_self = keys_self + keys_mask_self,
                                keys_copy = keys_copy,
                                keys_self_matrices = keys_mask_self)
        
        env = SpoofEntityWrapper(env, np.max(n_boxes), ['box_obs', 'you_lock', 'team_lock', 'obj_lock'], ['mask_ab_obs'])
    
        env = SpoofEntityWrapper(env, n_flags, ['food_obs'], ['mask_af_obs'])
        keys_mask_external += ['mask_ab_obs_spoof', 'mask_af_obs_spoof']
        
        env = LockAllWrapper(env,
                             remove_object_specific_lock = True)
        
        env = MaskActionWrapper(env,
                                action_key = 'action_pull',
                                mask_keys = ['mask_ab_obs', 'mask_ar_obs'])

        env = GrabClosestWrapper(env)
        
        env = DiscardMujocoExceptionEpisodes(env)

        env = ConcatenateObsWrapper(env, {'agent_qpos_qvel': ['agent_qpos_qvel', 'hider', 'prep_obs'],
                                        'box_obs': ['box_obs', 'you_lock', 'team_lock', 'obj_lock'],
                                        'ramp_obs': ['ramp_obs'] + (['ramp_you_lock', 'ramp_team_lock', 'ramp_obj_lock'] if lock_ramp else [])})
        
        env = SelectKeysWrapper(env, keys_self=keys_self,
                                keys_other=keys_external + keys_mask_self + keys_mask_external)

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
def object_placement(grid, obj_size, metadata, random_state, bounds = None):
    if bounds == None:
        return uniform_placement(grid, obj_size, metadata, random_state)

    pos = np.array([random_state.randint(bounds[0][0], bounds[1][0]),
                    random_state.randint(bounds[0][1], bounds[1][1])])
    return pos

'''
Makes the environment.
'''
def make_env():
    env_generator = GameEnvironment(lidar = 0)
    return env_generator.make_env()
