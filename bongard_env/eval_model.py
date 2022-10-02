import gym
import random
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import gridspec

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.manifold import TSNE



def plot_bp_ranking(env):
	"""Plots the 9 BPs the agent best performed on
	
	Args:
	    env (gym.Env): environment on which the evaluation was run
	"""
	bp_success = list(sorted(env.bp_success.items(), key=lambda item: item[1], reverse=True))

	f = plt.figure()
	rows, cols = 10, 10
	rows, cols = 3, 3
	for i, bp in enumerate(bp_success):

		bp_path = "BPs/" + bp[0] + "/" + bp[0] + ".gif" 
		img = Image.open(bp_path).convert('1')
		img = np.asarray(img)

		ax = f.add_subplot(rows,cols, i+1)
		ax.annotate(bp[1], (300,0), fontsize=20)
		ax.annotate(bp[0], (10,0), fontsize=20)
		ax.axis('off')
		plt.imshow(img, cmap='Greys_r')

		if i > 7:
			break
	
	# f.suptitle("Reward Ranking for each BP")
	# plt.savefig(f'bp_ranking.pdf', dpi=300, bbox='tight')

	plt.show(block=True)



def plot_feature_space(env, info_dict):
	"""Plots a 2d-plot of the feature space for the image comparisons
	
	Args:
	    env (gym.Env): the environment on which the evaluation was run
	    info_dict (dict): Output dict containing stats about the evaluation
	"""
	bp_success = list(sorted(env.bp_success.items(), key=lambda item: item[1], reverse=True))


	features = np.array(info_dict['features'])


	pca = sklearnPCA(n_components=2)
	# transformed = pca.fit_transform(info_dict['features'])
	transformed2 = pca.fit_transform(features)


	tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=100)
	transformed = tsne.fit_transform(features)

	bp_class = np.array(list(map(lambda x: int(x[1:]), info_dict['bp_class'])))
	same_classes = np.array(info_dict['same_classes'])
	pairs = np.array(info_dict['pairs'])
	actions = np.array(info_dict['actions'])
	corrects = (same_classes == actions).astype(int)
	for i in range(len(same_classes)):

		if actions[i] == 0:
			corrects[i] = 0
		elif actions[i] == same_classes[i]:
			corrects[i] = 1
		elif actions[i] != same_classes[i]:
			corrects[i] = 2

	top25bp = [int(x[0][1:]) for x in bp_success[:25]]
	top10 = top25bp[:9]

	x = transformed[:,0]
	y = transformed[:,1]

	x2 = transformed2[:,0]
	y2 = transformed2[:,1]

	f = plt.figure()
	rows, cols = 1, 1

	ax = f.add_subplot(rows,cols, 1)
	for bp in top10:
		ax.scatter(x[bp == bp_class], y[bp == bp_class], label=bp)
	ax.legend(loc='upper right')

	plt.show()


def eval_model(model, env, feature_space='diff', n_eps=50, render=False):
	"""A function for running an evaluation for a trained agent
	on an environment
	
	Args:
	    model (TYPE): trained model
	    env (gym.Env): initialized environment
	    feature_space (str, optional): the type of feature space to log
	    n_eps (int, optional): number of episodes for evaluation
	    render (bool, optional): whether to render the evaluation
	
	Returns:
	    gym.Env, dict: environment on which evaluation was run and dict
	    			   with logged information of the run
	"""
	observation = env.reset()

	info_dict = {'bp_class' : [], 'features' : [], 'pairs' : [], 'same_classes' : [], 'actions' : [],}


	last_reward = 1
	for ep in range(n_eps):

		done = False

		while not done:

			action, _, feature = model.predict(observation, deterministic=True)

			observation, reward, done, info = env.step(action)


			current_bp = env.bp_paths[env.bp_i]
			same_class = env.current_bp_pair[-1]

			info_dict['pairs'].append(env.current_bp_pair)

			label_map = {'a' : 0, 'b':1}

			if render and env.reward == 0 or (render and last_reward == 0):
				env.render()

			last_reward = reward

			if feature_space == 'single':
				info_dict['features'].append(np.array(feature[1][0].squeeze()))
				info_dict['features'].append(np.array(feature[1][1].squeeze()))
				bp_class += [current_bp, current_bp]

				info_dict['same_classes'].append(label_map[env.current_bp_pair[0][-6]])
				info_dict['same_classes'].append(label_map[env.current_bp_pair[1][-6]])

			elif feature_space == 'diff':
				info_dict['features'].append(np.array(feature[0].squeeze()))
				info_dict['bp_class'].append(current_bp)
				info_dict['same_classes'].append(same_class)
				info_dict['actions'].append(action)

			if done:
				observation = env.reset()

	return env, info_dict

def dict_to_runname(params):
    """helper function for turning a dict of hyperparameters
    into a string
    
    Args:
        params (dict): Dictionary of hyperparameters
    
    Returns:
        str: string describing hyperparameters
    """
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