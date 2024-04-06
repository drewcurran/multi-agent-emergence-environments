import os
from os.path import join
import numpy as np
import tensorflow as tf
import logging
import sys
import traceback
import cloudpickle as pickle
from ma_policy.ma_policy import MAPolicy
from mujoco_worldgen.util.envs import get_match


def shape_list(x):
    '''
        deal with dynamic shape in tensorflow cleanly
    '''
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]


def replace_base_scope(var_name, new_base_scope):
    split = var_name.split('/')
    split[0] = new_base_scope
    return os.path.normpath('/'.join(split))


def load_variables(policy, weights):
    weights = {os.path.normpath(key): value for key, value in weights.items()}
    weights = {replace_base_scope(key, policy.scope): value for key, value in weights.items()}
    assign_ops = []
    for var in policy.get_variables():
        var_name = os.path.normpath(var.name)
        if var_name not in weights:
            logging.warning(f"{var_name} was not found in weights dict. This will be reinitialized.")
            tf.get_default_session().run(var.initializer)
        else:
            try:
                assert np.all(np.array(shape_list(var)) == np.array(weights[var_name].shape))
                assign_ops.append(var.assign(weights[var_name]))
            except Exception:
                traceback.print_exc(file=sys.stdout)
                print(f"Error assigning weights of shape {weights[var_name].shape} to {var}")
                sys.exit()
    tf.get_default_session().run(assign_ops)


def load_policy(pattern, core_dir='', pols_dir='examples', exact=False, env=None, scope='policy'):
    '''
        Load a policy.
        Args:
            path (string): policy path
            env (Gym.Env): This will update the observation space of the
                policy that is returned
            scope (string): The base scope for the policy variables
    '''
    # Loads environment from generic file
    if not os.path.exists(pattern):
        match = get_match(parent = join(core_dir, pols_dir),
                          pattern = pattern,
                          file = os.path.basename(pattern) if exact else '*',
                          file_types = ['', '.npz'])
        return load_policy(match, env=env)

    # Loads environment from python file
    if pattern.endswith(".npz"):
        print("Loading policy from the file: %s" % pattern)

    # TODO this will probably need to be changed when trying to run policy on GPU
    if tf.get_default_session() is None:
        tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1)
        sess = tf.Session(config=tf_config)
        sess.__enter__()

    # Load the policy from the filepath
    policy_dict = dict(np.load(pattern))
    policy_fn_and_args_raw = pickle.loads(policy_dict['policy_fn_and_args'])
    policy_args = policy_fn_and_args_raw['args']
    policy_args['scope'] = scope

    # Use the observation and action space of the environment
    if env is not None:
        policy_args['ob_space'] = env.observation_space
        policy_args['ac_space'] = env.action_space

    # Build the policy
    policy = MAPolicy(**policy_args)
    del policy_dict['policy_fn_and_args']

    # Load the weights for the policy
    load_variables(policy, policy_dict)

    return policy
