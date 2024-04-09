#!/usr/bin/env python3.6
import logging
import click
from os.path import abspath, dirname, join
from mujoco_worldgen.util.parse_arguments import parse_arguments
from ma_policy.ma_policy2 import MAPolicy
from load_modules.load_env import load_env
from load_modules.load_policy import load_policy
from mujoco_worldgen.util.types import extract_matching_arguments

logger = logging.getLogger(__name__)

@click.command()
@click.argument('argv', nargs=-1, required=False)
def main(argv):
    '''
        Train policy on unseeded environments.
    '''
    core_dir = abspath(join(dirname(__file__), '..'))
    envs_dir = 'mae_envs/envs'
    pols_dir = 'ma_policy/pols'

    names, kwargs = parse_arguments(argv)

    env_name = names[0]
    env, args_remaining_env = load_env(env_name, core_dir, envs_dir, **kwargs)
    env.reset()
    
    # Examine the environment
    if len(names) == 1:
        print("Training new model.")
        assert len(args_remaining_env) == 0
        policy = MAPolicy(env)

    # Run policies on the environment
    if len(names) == 2:  
        print("Training old model.")
        pol_name = names[1]
        policy = load_policy(pol_name, core_dir, pols_dir, env=env, scope="policy", **kwargs)
    
    if kwargs['load_weights']:
        policy.policy_net.load_weights(f'ma_policy/pols/capture_the_flag/policy{policy.episodes}')
        policy.value_net.load_weights(f'ma_policy/pols/capture_the_flag/values{policy.episodes}')

    args_to_pass, args_remaining_viewer = extract_matching_arguments(MAPolicy.train, kwargs)
    args_remaining = set(args_remaining_env).intersection(set(args_remaining_viewer))
    assert len(args_remaining) == 0

    policy.reset()
    policy.train(**args_to_pass, episodes=1)

    policy.policy_net.save(f'ma_policy/pols/capture_the_flag/policy{policy.episodes}')
    policy.value_net.save(f'ma_policy/pols/capture_the_flag/values{policy.episodes}')

    print(main.__doc__)


if __name__ == '__main__':
    logging.getLogger('').handlers = []
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    main()
