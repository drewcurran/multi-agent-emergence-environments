# Multiagent emergence environments
Training policies and viewing environments adapted from Emergent Tool Use From Multi-Agent Autocurricula ([blog](https://openai.com/blog/emergent-tool-use/), [paper](https://arxiv.org/abs/1909.07528)).

## Installation
### Dependencies
```
pip install -r mujoco-worldgen/requirements.txt
pip install -e mujoco-worldgen/
pip install -e multi-agent-emergence-environments/
pip install -r multi-agent-emergence-environments/requirements_ma_policy.txt
```
### Hardware
```
macOS
Python 3.6
```

## Environment Generation
The baseline environment is: \
*Hide-and-seek* - `mae_envs/envs/py/hide_and_seek.py` - The environment studied in the paper. \
The new environment is: \
*Capture-the-flag* - `mae_envs/envs/py/capture_the_flag.py` - The environment with which transfer learning is applied. \
The `bin/examine` script displays environments and can also play saved policies:
```
bin/examine.py hide_and_seek
bin/examine.py hide_and_seek_quadrant hide_and_seek_quadrant
``` 

## Policy Optimization
WIP
