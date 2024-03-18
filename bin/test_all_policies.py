#!/usr/bin/env python3.6
import subprocess
import unittest
import pytest
import os
from os.path import abspath, dirname, join


CORE_DIRECTORY = abspath(join(dirname(__file__), '..'))
ENVS_DIRECTORY = 'mae_envs/envs'
POLS_DIRECTORY = 'ma_policy/pols'
EXAMINE_DIRECTORY = 'bin/examine.py'


class ExamineTest(unittest.TestCase):
    def test_examine_env(self):
        envs = [
            "hide_and_seek_full.jsonnet",
            "hide_and_seek_quadrant.jsonnet",
            "blueprint.jsonnet",
            "lock_and_return.jsonnet",
            "sequential_lock.jsonnet",
            "shelter.jsonnet",
        ]
        for env in envs:
            with self.assertRaises(subprocess.TimeoutExpired):
                subprocess.check_call(
                    [
                        "/usr/bin/env",
                        "python",
                        join(CORE_DIRECTORY, EXAMINE_DIRECTORY),
                        join(CORE_DIRECTORY, ENVS_DIRECTORY, env)
                    ],
                    timeout=10)


    def test_examine_policies(self):
        envs_policies = [
            ("hide_and_seek_full.jsonnet", "hide_and_seek_full.npz"),
            ("hide_and_seek_quadrant.jsonnet", "hide_and_seek_quadrant.npz"),
            ("blueprint.jsonnet", "blueprint.npz"),
            ("lock_and_return.jsonnet", "lock_and_return.npz"),
            ("sequential_lock.jsonnet", "sequential_lock.npz"),
            ("shelter.jsonnet", "shelter.npz"),
        ]
        for env, policy in envs_policies:
            with self.assertRaises(subprocess.TimeoutExpired):
                subprocess.check_call(
                    [
                        "/usr/bin/env",
                        "python",
                        join(CORE_DIRECTORY, EXAMINE_DIRECTORY),
                        join(CORE_DIRECTORY, ENVS_DIRECTORY, env),
                        join(CORE_DIRECTORY, POLS_DIRECTORY, policy)
                    ],
                    timeout=15)
