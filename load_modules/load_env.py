import json
import _jsonnet
from runpy import run_path
from mujoco_py import load_model_from_xml, load_model_from_mjb, MjSim
from mujoco_worldgen import Env
from mujoco_worldgen.util.types import extract_matching_arguments
from mujoco_worldgen.parser import parse_file, unparse_dict
from load_modules.flexible_load import load_file


def load_py(pattern, **kwargs):
    '''
        Loads environment from python file
    '''
    print("Loading environment from the file: %s" % pattern)
    module = run_path(pattern)
    make_env = module["make_env"]
    args_to_pass, args_remaining = extract_matching_arguments(make_env, kwargs)
    env = make_env(**args_to_pass)
    return env, args_remaining

def load_jsonnet(pattern, **kwargs):
    '''
        Loads environment from jsonnet file
    '''
    print("Loading environment from the file: %s" % pattern)
    env_data = json.loads(_jsonnet.evaluate_file(pattern))
    def get_function(fn_data):
        name = fn_data['function']
        extra_args = fn_data['args']
        module_path, function_name = name.rsplit(':', 1)
        result = getattr(__import__(module_path, fromlist=(function_name,)), function_name)
        if len(extra_args) > 0:
            def result_wrapper(*args, **kwargs):
                actual_kwargs = extra_args.copy()
                actual_kwargs.update(kwargs)
                return result(*args, **actual_kwargs)
            return result_wrapper
        else:
            return result
    make_env = get_function(env_data['make_env'])
    args_to_pass, args_remaining = extract_matching_arguments(make_env, kwargs)
    env = make_env(**args_to_pass)
    return env, args_remaining

def load_xml(pattern, **kwargs):
    '''
        Loads environment from XML file
    '''
    print("Loading environment from the file: %s" % pattern)
    if len(kwargs) != 0:
        raise Exception("XML doesn't accept any extra input arguments")
    def get_sim(seed):
        xml_dict = parse_file(pattern, enforce_validation=False)
        xml = unparse_dict(xml_dict)
        model = load_model_from_xml(xml)
        return MjSim(model)
    env = Env(get_sim=get_sim)
    return env

def load_mjb(pattern, **kwargs):
    '''
        Loads environment from mjb file
    '''
    print("Loading environment from the file: %s" % pattern)
    if len(kwargs) != 0:
        raise Exception("MJB doesn't accept any extra input arguments")
    def get_sim(seed):
        model = load_model_from_mjb(pattern)
        return MjSim(model)
    env = Env(get_sim=get_sim)
    return env

def load_env(env_name, core_dir, envs_dir, **kwargs):
    '''
        Loads environment from directory.
    '''
    file_types = {'.py': load_py, '.jsonnet': load_jsonnet, '.xml': load_xml, '.mjb': load_mjb}
    env = load_file(env_name, core_dir=core_dir, sub_dir=envs_dir, file_types=file_types, **kwargs)
    if env is None:
        raise Exception(f'Could not find environment based on pattern {env_name}')
    return env
