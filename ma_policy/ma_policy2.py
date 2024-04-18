import tensorflow as tf
import numpy as np
from copy import deepcopy
from baselines.common.distributions import make_pdtype 
from tensorflow.keras.layers import Dense, LayerNormalization
from ma_policy.layers2 import (EMANormalization, CircularConv1D, ResidualSABlock, LSTM,
                               FlattenOuter, Concatenate, EntityConcatenate, Pooling)

class MAPolicy:
    '''
        Model for the policy.
    '''
    def __init__(self, env, stochastic=True, normalize=True):
        self.env = env
        self.ob_space = deepcopy(env.observation_space)
        self.ac_space = deepcopy(env.action_space)
        self.stochastic = stochastic
        self.normalize = normalize
        
        self.policy_net = PolicyNetwork(self.configure_actions())
        self.value_net = ValueNetwork()

        self.episodes = 0
        self.losses = []
    
    def configure_actions(self):
        self.pdtypes = {k: make_pdtype(s.spaces[0]) for k, s in self.ac_space.spaces.items()}
        action_shapes = {}
        for k, pdtype in self.pdtypes.items():
            total_params = pdtype.param_shape()
            shape = pdtype.sample_shape()
            action_shapes[k] = total_params, shape
        return action_shapes

    def train(self, episodes=100, lr=3e-4, gamma=0.99, clip_range=0.2, old_probs=None, save=100, filepath=''):
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        for episode in range(episodes):
            trajectories = self.collect_trajectories()
            while trajectories is None:
                trajectories = self.collect_trajectories()
            
            observations, actions, returns, advantages = self.compute_advantages_and_returns(trajectories, gamma)
            old_probs = self.update_policy_and_value(observations, actions, returns, advantages, old_probs, clip_range)

            print(f"Episode {self.episodes + 1} ({episode + 1}) complete.")
            if self.episodes % save == 0:
                print('Saving networks and statistics')
                self.policy_net.save_weights(f'{filepath}/policy{self.episodes}')
                self.value_net.save_weights(f'{filepath}/values{self.episodes}')
                np.save(f'{filepath}/losses{self.episodes}', np.array(self.losses))
            self.episodes += 1

    def collect_trajectories(self):
        obs = self.env.reset()
        trajectories = []
        while True:
            actions = self.policy_net(obs)

            chosen = {}
            for action_type, outputs in actions.items():
                chosen[action_type] = np.zeros((outputs.shape[1], outputs.shape[0]), dtype=np.int64)
                for a in range(outputs.shape[1]):
                    for s in range(outputs.shape[0]):
                        chosen[action_type][a][s] = np.random.choice(len(outputs[s][a]), p=outputs[s][a])
                if outputs.shape[0] == 1:
                    chosen[action_type] = np.squeeze(chosen[action_type], 1)
            
            try:
                new_obs, reward, done, _ = self.env.step(chosen)
            except:
                self.env.reset()
                self.policy_net.reset()
                return None
                
            trajectories.append((obs, chosen, reward))

            obs = new_obs
            if done:
                self.env.reset()
                self.policy_net.reset()
                break
        
        return trajectories
    
    def compute_advantages_and_returns(self, trajectories, gamma):
        observations, actions, rewards = zip(*trajectories)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.append(G)
        returns.reverse()
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        for obs in observations:
            value_estimates = self.value_net(obs)
        advantages = returns - tf.squeeze(value_estimates)
        return observations, actions, returns, advantages

    def update_policy_and_value(self, observations, actions, returns, advantages, old_action_probs, clip_range):
        policy_loss = 0
        value_loss = 0

        with tf.GradientTape(persistent=True) as tape:
            new_action_probs = []
            for i in range(len(observations)):
                new_action_probs.append(self.policy_net(observations[i]))
                new_value_estimate = tf.squeeze(self.value_net(observations[i]))

            if old_action_probs is None:
                return new_action_probs
            
            for i in range(len(observations)):
                for action_type in new_action_probs[i].keys():
                    new_action_prob = new_action_probs[i][action_type]
                    old_action_prob = old_action_probs[i][action_type]

                    ratio = new_action_prob / old_action_prob
                    surrogate_loss_1 = ratio * advantages[i][0]
                    surrogate_loss_2 = tf.clip_by_value(ratio, 1 - clip_range, 1 + clip_range) * advantages[i][0]
                    policy_loss += -tf.reduce_mean(tf.minimum(surrogate_loss_1, surrogate_loss_2))
                    value_loss += tf.keras.losses.mean_squared_error(returns[i], new_value_estimate)

        self.losses += [policy_loss, value_loss]

        policy_grads = tape.gradient(policy_loss, self.policy_net.trainable_variables)
        value_grads = tape.gradient(value_loss, self.value_net.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy_net.trainable_variables))
        self.value_optimizer.apply_gradients(zip(value_grads, self.value_net.trainable_variables))

        return new_action_probs

    def process_output(self, output):
        if isinstance(output, dict):
            return {k: np.squeeze(v, 1) for k, v in output.items()}
        else:
            return np.squeeze(output, 1)
        
    def reset(self):
        self.policy_net.reset()
        self.value_net.reset()


class LayerWrapper:
    def __init__(self, layer, inputs, outputs, inputs_dict={}, outputs_dict={}):
        self.layer = layer
        self.node_inputs = inputs
        self.node_outputs = outputs
        self.inputs_dict = inputs_dict
        self.outputs_dict = outputs_dict
        if len(self.outputs_dict) > 0:
            if isinstance(self.node_outputs, list):
                self.node_outputs += list(self.outputs_dict.values())
            else:
                self.node_outputs = [self.node_outputs] + list(self.outputs_dict.values())
    
    def call(self, values):
        inputs_dict = {}
        for key, val in self.inputs_dict.items():
            if isinstance(val, (list, tuple)):
                inputs_dict[key] = [values[v] if v is not None else None for v in val]
            else:
                inputs_dict[key] = values[val]
        if isinstance(self.node_inputs, (list, tuple)):
            out = self.layer([values[node] for node in self.node_inputs], **inputs_dict)
        else:
            out = self.layer(values[self.node_inputs], **inputs_dict)
        if isinstance(self.node_outputs, (list, tuple)):
            values.update({node: out[i] for i, node in enumerate(self.node_outputs)})
        else:
            values[self.node_outputs] = out


class PolicyNetwork(tf.keras.Model):
    def __init__(self, action_shapes):
        super(PolicyNetwork, self).__init__()

        self.inputs_net = InputsNetwork()
        
        self.network = []
        self.network += [LayerWrapper(CircularConv1D(9, 3, activation='relu'), 'lidar', 'lidar')]
        self.network += [LayerWrapper(FlattenOuter(), 'lidar', 'lidar')]
        self.network += [LayerWrapper(Concatenate(), ['observation_self', 'lidar'], 'main')]
        self.network += [LayerWrapper(Concatenate(), ['agent_qpos_qvel', 'main'], 'agent_qpos_qvel')]
        self.network += [LayerWrapper(Concatenate(), ['box_obs', 'main'], 'box_obs')]
        self.network += [LayerWrapper(Concatenate(), ['ramp_obs', 'main'], 'ramp_obs')]
        self.network += [LayerWrapper(Concatenate(), ['flag_obs', 'main'], 'flag_obs')]
        self.network += [LayerWrapper(Dense(128, activation='relu'), 'agent_qpos_qvel', 'agent_qpos_qvel')]
        self.network += [LayerWrapper(Dense(128, activation='relu'), 'box_obs', 'box_obs')]
        self.network += [LayerWrapper(Dense(128, activation='relu'), 'ramp_obs', 'ramp_obs')]
        self.network += [LayerWrapper(Dense(128, activation='relu'), 'flag_obs', 'flag_obs')]
        self.network += [LayerWrapper(Dense(128, activation='relu'), 'main', 'main')]
        self.network += [LayerWrapper(EntityConcatenate(-2), ['agent_qpos_qvel', 'box_obs', 'ramp_obs', 'flag_obs', 'main'], 'objects',
                                        {'masks_in': ['mask_aa_obs', 'mask_ab_obs', 'mask_ar_obs', 'mask_af_obs', None]},
                                        {'mask_out': 'objects_mask'})]
        self.network += [LayerWrapper(ResidualSABlock(4, 128), 'objects', 'objects',
                                        {'attention_mask': 'objects_mask'})]
        self.network += [LayerWrapper(Pooling('avg_pooling'), 'objects', 'objects',
                                        {'mask': 'objects_mask'})]
        self.network += [LayerWrapper(Concatenate(), ['main', 'objects'], 'main')]
        self.network += [LayerWrapper(Dense(256, activation='relu'), 'main', 'main')]
        self.network += [LayerWrapper(LSTM(256), 'main', 'main',
                                        {'states': 'states'},
                                        {'states': 'states'})]
        self.network += [LayerWrapper(LayerNormalization(), 'main', 'main')]

        self.actions = {}
        for action_type, space in action_shapes.items():
            num_layers = space[1][0] if len(space[1]) == 1 else 1
            num_params = space[0][0] / num_layers
            for ct in range(num_layers):
                self.network += [LayerWrapper(Dense(num_params, activation='softmax'), 'main', f'{action_type}{ct}')]
            self.actions[action_type] = ct + 1

    def call(self, inputs):
        values = self.process_inputs(inputs)
        values['states'] = self.states
        for layer in self.network:
            layer.call(values)
        actions = {}
        for action_type, ct in self.actions.items():
            actions[action_type] = np.squeeze([values[f'{action_type}{i}'] for i in range(ct)], 2)
        return actions
    
    def reset(self):
        self.states = tf.zeros((2, 4, 256))

    def process_inputs(self, inputs):
        processed_inputs = self.inputs_net.call(inputs)
        for k, v in inputs.items():
            processed_inputs[k] = tf.expand_dims(v, 1)
        return processed_inputs
        

class ValueNetwork(tf.keras.Model):
    def __init__(self):
        super(ValueNetwork, self).__init__()

        self.inputs_net = InputsNetwork()

        self.network = []
        self.network += [LayerWrapper(CircularConv1D(9, 3, activation='relu'), 'lidar', 'lidar')]
        self.network += [LayerWrapper(FlattenOuter(), 'lidar', 'lidar')]
        self.network += [LayerWrapper(Concatenate(), ['observation_self', 'lidar'], 'main')]
        self.network += [LayerWrapper(Concatenate(), ['agent_qpos_qvel', 'main'], 'agent_qpos_qvel')]
        self.network += [LayerWrapper(Concatenate(), ['box_obs', 'main'], 'box_obs')]
        self.network += [LayerWrapper(Concatenate(), ['ramp_obs', 'main'], 'ramp_obs')]
        self.network += [LayerWrapper(Concatenate(), ['flag_obs', 'main'], 'flag_obs')]
        self.network += [LayerWrapper(Dense(128, activation='relu'), 'agent_qpos_qvel', 'agent_qpos_qvel')]
        self.network += [LayerWrapper(Dense(128, activation='relu'), 'box_obs', 'box_obs')]
        self.network += [LayerWrapper(Dense(128, activation='relu'), 'ramp_obs', 'ramp_obs')]
        self.network += [LayerWrapper(Dense(128, activation='relu'), 'flag_obs', 'flag_obs')]
        self.network += [LayerWrapper(Dense(128, activation='relu'), 'main', 'main')]
        self.network += [LayerWrapper(EntityConcatenate(-2), ['agent_qpos_qvel', 'box_obs', 'ramp_obs', 'flag_obs', 'main'], 'objects',
                                        {'masks_in': [None, None, None, None, None]},
                                        {'mask_out': 'objects_mask'})]
        self.network += [LayerWrapper(ResidualSABlock(4, 128), 'objects', 'objects',
                                        {'attention_mask': 'objects_mask'})]
        self.network += [LayerWrapper(Pooling('avg_pooling'), 'objects', 'objects',
                                        {'mask': 'objects_mask'})]
        self.network += [LayerWrapper(Concatenate(), ['main', 'objects'], 'main')]
        self.network += [LayerWrapper(Dense(256, activation='relu'), 'main', 'main')]
        self.network += [LayerWrapper(LSTM(256), 'main', 'main',
                                        {'states': 'states'},
                                        {'states': 'states'})]
        self.network += [LayerWrapper(LayerNormalization(), 'main', 'main')]
        self.network += [LayerWrapper(Dense(1), 'main', 'value')]
    
    def call(self, inputs):
        values = self.process_inputs(inputs)
        values['states'] = self.states
        for layer in self.network:
            layer.call(values)
        return values['value']

    def reset(self):
        self.states = tf.zeros((2, 4, 256))
    
    def process_inputs(self, inputs):
        processed_inputs = self.inputs_net.call(inputs)
        for k, v in inputs.items():
            processed_inputs[k] = np.expand_dims(v, 1)
        return processed_inputs


class InputsNetwork:
    def __init__(self):
        self.network = []
        self.network += [LayerWrapper(EMANormalization(), 'observation_self', 'observation_self')]
        self.network += [LayerWrapper(EMANormalization(), 'lidar', 'lidar')]
        self.network += [LayerWrapper(EMANormalization(), 'agent_qpos_qvel', 'agent_qpos_qvel')]
        self.network += [LayerWrapper(EMANormalization(), 'box_obs', 'box_obs')]
        self.network += [LayerWrapper(EMANormalization(), 'ramp_obs', 'ramp_obs')]
        self.network += [LayerWrapper(EMANormalization(), 'flag_obs', 'flag_obs')]
    
    def call(self, inputs):
        values = {k: inputs[k] for k in inputs.keys()}
        for layer in self.network:
            layer.call(values)
        return values
