import gym
import torch
import numpy as np
import random

from bongard_base_env import BPEnv

from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

log_dir = "./train_logs/"

algo_dict = {'PPO' : PPO, 'A2C' : A2C}

params = {
			'algo' : 'PPO',
			'policy' : 'MlpPolicy',
			'lr' : 7e-5,
			'eplength' : False,
			'CB' : False, 
			'nostate' : False, 
			'labelstate' : False, 
			'flatstate' : False,
			'smallstate' : False,
			'unbalanced' : False

		 }

param_list = []
for key, value in params.items():

	if type(value) == bool:
		if value:
			param_list.append(key)
	elif type(value) == int or type(value) == float:
		param_list.append(key + str(value))
	else:
		param_list.append(value)


seeds = [132, 357, 2, 466, 899]
run_name = '_'.join(param_list)

for seed in seeds:

	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	
	env = BPEnv(
		no_state=params['nostate'], 
		label_in_state=params['labelstate'], 
		flat_state=params['flatstate'], 
		small_state=params['smallstate'], 
		unbalanced=params['unbalanced'],
		ep_length=params['eplength'])

	model = algo(params['policy'], env, learning_rate=params['lr'], verbose=1, causal=params['CB'], tensorboard_log=f"./train_logs/{run_name}/")
	model.learn(total_timesteps=100000, tb_log_name=f"{run_name}_{seed}")











