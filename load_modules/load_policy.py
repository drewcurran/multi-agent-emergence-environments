import tensorflow as tf
import numpy as np
import cloudpickle as pickle
from load_modules.flexible_load import load_file
from ma_policy.ma_policy2 import MAPolicy
    

def load_npz(pattern, **kwargs):
    '''
        Loads policy from numpy file
    '''
    print("Loading policy from the file: %s" % pattern)
    from ma_policy.ma_policy import MAPolicy
    import os
    os.environ['LOG_NETS'] = 'False'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.compat.v1.disable_eager_execution()
    if tf.compat.v1.get_default_session() is None:
        tf_config = tf.compat.v1.ConfigProto(
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1)
        sess = tf.compat.v1.Session(config=tf_config)
        sess.__enter__()
    env = kwargs.pop('env', None)
    scope = kwargs.pop('scope', 'policy')
    policy_dict = dict(np.load(pattern))
    policy_args = pickle.loads(policy_dict.pop('policy_fn_and_args'))['args']
    policy_args['scope'] = scope
    policy_args['pi_network_spec'] = policy_args.pop('network_spec')
    if env is not None:
        policy_args['ob_space'] = env.observation_space
        policy_args['ac_space'] = env.action_space
    policy = MAPolicy(**policy_args)
    policy.load_weights_dict(policy_dict)
    return policy

def load_ckpt(pattern, **kwargs):
    '''
        Loads policy from ckpt file
    '''
    print("Loading policy from the file: %s" % pattern)
    env = kwargs.pop('env', None)
    policy = MAPolicy(env)
    policy.load_weights(pattern)
    return policy

def load_keras(pattern, **kwargs):
    '''
        Loads policy from keras file
    '''
    print("Loading policy from the file: %s" % pattern)
    policy = tf.keras.models.load_model(pattern)
    return policy

def load_policy(pol_name, core_dir, pols_dir, env=None, scope='policy', **kwargs):
    '''
        Loads policy from directory.
    '''
    file_types = {'.npz': load_npz, '.ckpt': load_ckpt, '.keras': load_keras}
    pol = load_file(pol_name, core_dir=core_dir, sub_dir=pols_dir, file_types=file_types, env=env, scope=scope, **kwargs)
    if pol is None:
        raise Exception(f'Could not find environment based on pattern {pol_name}')
    return pol
