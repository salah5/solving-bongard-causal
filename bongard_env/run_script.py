import gym
import torch
import numpy as np
import random
import sys
from PIL import Image
import matplotlib.pyplot as plt
from bongard_base_env import BPEnv
from stable_baselines3 import A2C, DQN, PPO
from eval_model import eval_model


def dict_to_runname(params):

    param_list = []
    for key, value in params.items():

        if type(value) == bool:
            if value:
                param_list.append(key)
        elif type(value) == int or type(value) == float:
            param_list.append(key + str(value))
        elif type(value) == list:
            continue
        elif key in ["env", "algo", "policy", ]:
            param_list.append(value)
        else:
            continue

    run_name = '_'.join(param_list)

    return run_name


algo_dict = {'PPO': PPO, 'A2C': A2C}
env_dict = {'BPEnv': BPEnv}

params = {

    'env': 'BPEnv2',
    'algo': 'PPO',
    'policy': 'SiaMlpPolicy',
    'lr': 7e-05,
    'eplength': False,
    'CB': False,
    'nostate': False,
    'labelstate': False,
    'flatstate': False,
    'smallstate': False,
    'unbalanced': False,
    'h_bounds': False,
    'clip_range': 0.2,
    'save_model': False,
    'seeds':  [11, 61, 331],
    'lrs': [1e-05, 3e-05, 4e-05, 5e-05, 7e-05, 9e-05],
    'clip_ranges': [0.2, 0.1, 0.05, 0.4],
    'log_dir': 'logs/',
    'total_timesteps': 2000000,
    'skip_action' : True,
    'test_mode': False,
}


env = env_dict[params['env']](skip_action=params['skip_action'])


if params['test_mode']:

    torch.manual_seed(5)
    np.random.seed(5)
    random.seed(5)

    params['run_name'] = ""

    algo = algo_dict[params['algo']](params['policy'], env, learning_rate=params['lr'], clip_range=params['clip_range'], verbose=1, causal=params['CB'], tensorboard_log=f"./{params['log_dir']}/{params['run_name']}/")
    model = algo.load(params['run_name'])
    eval_model(model, env, feature_space='diff', n_eps=50, top25=False,
               bp_ranking=True, bp_feature_space=True, render=False)

    sys.exit()


for seed in params['seeds']:

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    params['seed'] = seed
    params['run_name'] = dict_to_runname(params)
    algo = algo_dict[params['algo']](params['policy'], env, learning_rate=params['lr'], clip_range=params['clip_range'], verbose=1, causal=params['CB'], tensorboard_log=f"./{params['log_dir']}/{params['run_name']}/")

    algo.learn(total_timesteps=params['total_timesteps'], tb_log_name=f"{params['run_name']}_{seed}")
    algo.save(params['run_name'])
