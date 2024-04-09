#!/usr/bin/env python3.6
import logging
import click
from os.path import abspath, dirname, join
from gym.spaces import Tuple
from mujoco_worldgen.util.types import extract_matching_arguments
from mujoco_worldgen.util.parse_arguments import parse_arguments
from mae_envs.viewer.env_viewer import EnvViewer
from mae_envs.wrappers.multi_agent import JoinMultiAgentActions
from mae_envs.viewer.policy_viewer import PolicyViewer
from load_modules.load_env import load_env
from load_modules.load_policy import load_policy

logger = logging.getLogger(__name__)


@click.command()
@click.argument('argv', nargs=-1, required=False)
def main(argv):
    '''
        Display unseeded environments and run policies.
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
        assert len(args_remaining_env) == 0

        viewer = EnvViewer(env)
        viewer.run()

    # Run policies on the environment
    if len(names) >= 2:  
        if isinstance(env.action_space, Tuple):
            env = JoinMultiAgentActions(env)

        pol_names = names[1:]
        policies = [load_policy(pol_name, core_dir, pols_dir, env=env, scope=f"policy_{i}", **kwargs) for i, pol_name in enumerate(pol_names)]

        args_to_pass, args_remaining_viewer = extract_matching_arguments(PolicyViewer, kwargs)
        args_remaining = set(args_remaining_env).intersection(set(args_remaining_viewer))
        assert len(args_remaining) == 0

        viewer = PolicyViewer(env, policies, **args_to_pass)
        viewer.run()

    print(main.__doc__)


if __name__ == '__main__':
    logging.getLogger('').handlers = []
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    main()
