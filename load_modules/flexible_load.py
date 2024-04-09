import os
from os.path import join
from glob import glob
import numpy as np


def get_match(pattern, dir, file_name, file_types, subdirectories='**'):
        if type(subdirectories) not in [list, tuple, np.ndarray]:
            subdirectories = [subdirectories] * len(file_types)
        matches = []
        for i in range(len(file_types)):
            path = join(dir, subdirectories[i], file_name + file_types[i])
            matches += glob(path, recursive=True)
        matches = [match for match in matches if match.find(pattern) > -1]
        matches = [match for match in matches if os.path.isfile(match)]
        matches = [match for match in matches if not os.path.basename(match).startswith('test_')]
        assert len(matches) < 2, "Found multiple files: %s" % str(matches)
        assert len(matches) > 0, "Found no files"
        return matches[0]


def get_path(pattern, dir, file_types, exact=False):
    # Loads environment from generic file
    if not os.path.exists(pattern):
        match = get_match(pattern = pattern,
                          dir = dir,
                          file_name = os.path.basename(pattern) if exact else '*',
                          file_types = file_types)
        return get_path(match, dir, file_types, exact = exact)
    
    return pattern


def load_file(pattern, core_dir, sub_dir, file_types, **kwargs):
    '''
    Flexible load of a file based on pattern.
    Args:
        pattern: tries to match file to the pattern.
        core_dir: Absolute path to the core code directory for the project.
        sub_dir: relative path (from core_dir) to folder containing files.
        file_types: dictionary of file types and their respective loading functions.
        kwargs: arguments passed to a function related to the file.
    '''
    dir = join(core_dir, sub_dir)
    search_types = [''] + list(file_types.keys())
    pattern = get_path(pattern, dir, search_types, exact = True)

    for type, load_fn in file_types.items():
        if pattern.endswith(type):
            return load_fn(pattern, **kwargs)
    return None
