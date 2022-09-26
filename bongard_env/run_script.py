import gym
import torch
import numpy as np
import random
import sys
from PIL import Image
import matplotlib.pyplot as plt

from bongard_base_env import BPEnv
from bongard_counter_env import BPEnv2

from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
import gym_bandits
from pprint import pprint
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


algo_dict = {'PPO': PPO, 'A2C': A2C, 'RPPO': RecurrentPPO}
env_dict = {'BPEnv': BPEnv, 'BPEnv2': BPEnv2}

params = {

    'env': 'BPEnv2',
    # 'algo' : 'RPPO',
    # 'policy' : 'MlpLstmPolicy',
    'algo': 'PPO',
    'policy': 'SiaMlpPolicy',
    # 'policy': 'SiaCnnPolicy',
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
    # 'run_name' : 'BPEnv2_PPO_SiaMlpPolicy_lr1e-05_clip_range0.2_27',
    # 'run_name' : 'BPEnv_PPO_SiaMlpPolicy_lr7e-05_clip_range0._n',
    # 'run_name' : 'BPEnv2_PPO_SiaMlpPolicy_lr1e-05_clip_range0.2_area',
    # 'run_name': dict_to_runname(params),
    # 'run_name': "final_run",
    'seeds':  [11, 61],# [999, 357, 2],  # , 357, 2]#, 466, 899]
    'lrs': [1e-05, 3e-05, 4e-05, 5e-05, 7e-05, 9e-05],
    'clip_ranges': [0.2, 0.1, 0.05, 0.4],
    # 'log_dir': 'bp2/',
    'log_dir': 'thesis_logs/cnn_vs_mlp/',
    'total_timesteps': 2000000,
    'skip_action' : True,
    'test_mode': True,
}

# params['run_name'] = dict_to_runname(params)

# env = env_dict[params['env']](eval=params['test_mode'])
env = env_dict[params['env']](skip_action=params['skip_action'])
params['run_name'] = "good models/BPEnv2_PPO_SiaMlpPolicy_lr1e-05_clip_range0.2_27"
# params['run_name'] = "BPEnv2_PPO_SiaMlpPolicy_lr1e-05_clip_range0.2_27"

if params['test_mode']:

    torch.manual_seed(5)
    np.random.seed(5)
    random.seed(5)

    algo = algo_dict[params['algo']](params['policy'], env, learning_rate=params['lr'], clip_range=params['clip_range'], verbose=1, causal=params['CB'], tensorboard_log=f"./{params['log_dir']}/{params['run_name']}/")
    model = algo.load(params['run_name'])
    eval_model(model, env, feature_space='diff', n_eps=50, top25=False,
               bp_ranking=True, bp_feature_space=True, render=False)

    sys.exit()


# for clip_range in params['clip_ranges']:

#     params['clip_range'] = clip_range

for seed in params['seeds']:

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    params['seed'] = seed
    params['run_name'] = dict_to_runname(params)
    params['run_name'] = params['run_name'].replace("CB", "CB2")
    algo = algo_dict[params['algo']](params['policy'], env, learning_rate=params['lr'], clip_range=params['clip_range'], verbose=1, causal=params['CB'], tensorboard_log=f"./{params['log_dir']}/{params['run_name']}/")

    algo.learn(total_timesteps=params['total_timesteps'], tb_log_name=f"{params['run_name']}_{seed}")
    algo.save(params['run_name'])
