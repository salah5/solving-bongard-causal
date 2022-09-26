import gym
import random
import os
import sys
import numpy as np
from gym import spaces
from PIL import Image
import itertools
import matplotlib.pyplot as plt
from matplotlib import gridspec

from stable_baselines3.common.evaluation import evaluate_policy
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE

from pprint import pprint

top25bps = ['p073', 'p051', 'p009', 'p078', 'p070', 
			'p008', 'p069', 'p031', 'p068', 'p023', 
			'p054', 'p061', 'p022', 'p043', 'p015', 
			'p085', 'p038', 'p036', 'p053', 'p020', 
			'p032', 'p065', 'p037', 'p096', 'p047']

top10 = ['p073', 
		 'p051', 
		 'p009', 
		 'p070', 
		 'p078', 
		 'p008', 
		 'p023', 
		 'p043', 
		 'p022', 
		 'p061']

def plot_bp_ranking(env, top25=False):

	bp_success = list(sorted(env.bp_success.items(), key=lambda item: item[1], reverse=True))

	print("TOP 10")
	print(bp_success[:10])

	f = plt.figure()
	rows, cols = 10, 10
	rows, cols = 3, 3
	for i, bp in enumerate(bp_success):

		bp_path = "BPs/" + bp[0] + "/" + bp[0] + ".gif" 
		img = Image.open(bp_path).convert('1')
		img = np.asarray(img)

		ax = f.add_subplot(rows,cols, i+1)
		ax.annotate(bp[1], (300,0))
		ax.annotate(bp[0], (10,0), fontsize=5)
		ax.axis('off')
		plt.imshow(img, cmap='Greys_r')

		if i > 7:
			break
	
	f.suptitle("Reward Ranking for each BP")
	plt.savefig(f'thesis_figures/BP good model ranking (top9).pdf', dpi=300, bbox='tight')


	plt.show(block=True)

def analyze_feature_space(features, bp_class, pairs, same_classes, actions, env):

	bp_success = list(sorted(env.bp_success.items(), key=lambda item: item[1], reverse=True))
	features = np.array(features)


	tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=100)
	transformed = tsne.fit_transform(features)

	x = transformed[:,0]
	y = transformed[:,1]
	bp_class = np.array(list(map(lambda x: int(x[1:]), bp_class)))


	top25bp = [int(x[0][1:]) for x in bp_success[:25]]
	top10 = top25bp[:10]
	og_pairs1 = np.array(pairs)[:,0]
	og_pairs2 = np.array(pairs)[:,1]
	pairs = [x for x in pairs if int(x[0][1:4]) in top10]

	top10_pairs = []
	for p in pairs:

		if int(p[0][1:4]) in top10:
			top10_pairs.append(p)

	for i in range(len(top10_pairs)):

		# if top10_pairs[i][0] != top10_pairs[i][1]:
		# 	continue

		xpairs = x[(og_pairs1 == top10_pairs[i][0]) & (og_pairs2 == top10_pairs[i][1])]
		ypairs = y[(og_pairs1 == top10_pairs[i][0]) & (og_pairs2 == top10_pairs[i][1])]

		# if not (np.any([xpairs < -30]) and np.any([ypairs < -50])):
		# 	continue

		f = plt.figure()
		rows, cols = 1, 3
		spec = gridspec.GridSpec(ncols=3, nrows=1,
                         width_ratios=[1, 1, 3])


		pos = 1
		for image in top10_pairs[i][:2]:
			img_path = "BPs/" + image[:4] + "/" + image
			ax = f.add_subplot(spec[pos-1])
			img = Image.open(img_path).convert('1')
			img = np.asarray(img)
			ax.imshow(img, cmap='Greys_r')
			pos += 1

		ax = f.add_subplot(spec[2])
		for bp in top10:
			print(bp)
			ax.scatter(x[bp == bp_class], y[bp == bp_class], label=bp, alpha=0.1)

		print(top10_pairs[i])
		print(bp_class[i])

		ax.scatter(x[(og_pairs1 == top10_pairs[i][0]) & (og_pairs2 == top10_pairs[i][1])],
				   y[(og_pairs1 == top10_pairs[i][0]) & (og_pairs2 == top10_pairs[i][1])], label=int(top10_pairs[i][0][1:4]))

			# filter = []
			# for k in range(len(og_pairs)):
			# 	if (og_pairs[k][0] == pairs[i][0]) and (og_pairs[k][1] == pairs[i][1]):
			# 		filter.append(True)
			# 	else:
			# 		filter.append(False)

			# filter = np.array(filter)
			# ax.scatter(x[filter], y[filter], label=bp, alpha=1)
			

		ax.legend(loc='upper right')

		plt.show(block=True)


def plot_image_space(features, bp_class, env):

	features = np.array(features)
	bp_class = np.array(list(map(lambda x: int(x[1:]), bp_class)))

	bp_success = list(sorted(env.bp_success.items(), key=lambda item: item[1], reverse=True))
	top25bp = [int(x[0][1:]) for x in bp_success[:25]]
	top10 = top25bp[:10]

	tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=100)
	transformed = tsne.fit_transform(features)

	x = transformed[:,0]
	y = transformed[:,1]

	f = plt.figure()
	rows, cols = 1, 1

	ax = f.add_subplot(rows,cols, 1)
	for bp in top10:
		ax.scatter(x[bp == bp_class], y[bp == bp_class], label=bp)
	ax.legend(loc='upper right')

	plt.show()


	sys.exit()


def plot_feature_space(features, bp_class, same_classes, actions, env, pairs):

	bp_success = list(sorted(env.bp_success.items(), key=lambda item: item[1], reverse=True))


	features = np.array(features)


	pca = sklearnPCA(n_components=2)
	# transformed = pca.fit_transform(features)
	transformed2 = pca.fit_transform(features)


	tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=100)
	transformed = tsne.fit_transform(features)

	bp_class = np.array(list(map(lambda x: int(x[1:]), bp_class)))
	same_classes = np.array(same_classes)
	pairs = np.array(pairs)
	actions = np.array(actions)
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


	# color_class = ['skip', 'same class', 'diferent class']
	# ax = f.add_subplot(rows,cols, 1)
	# for action in [0, 1, 2]:
	# 	ax.scatter(x[actions == action], y[actions == action], label=color_class[action])
	# 	# ax.scatter(x2[actions == action], y2[actions == action], label=color_class[action])
	# ax.legend(loc='upper right')


	# ax = f.add_subplot(rows,cols, 2)
	# color_class = ['not top25', 'top25']
	# top25mask = np.in1d(bp_class, top25bp).astype(int)
	# for action in [0, 1]:
	# 	ax.scatter(x[top25mask == action], y[top25mask == action], label=color_class[action])
	# ax.legend(loc='upper right')

	ax = f.add_subplot(rows,cols, 1)
	for bp in top10:
		ax.scatter(x[bp == bp_class], y[bp == bp_class], label=bp)
	ax.legend(loc='upper right')


	# ax = f.add_subplot(rows,cols, 3)
	# color_class = ['skip', 'correct', 'incorrect']
	# top25mask = np.in1d(bp_class, top25bp).astype(int)
	# print(top25mask)
	# for action in [0, 1, 2]:
	# 	ax.scatter(x[corrects == action], y[corrects == action], label=color_class[action])
	# ax.legend(loc='upper right')
	plt.savefig(f'thesis_figures/BP good model feature space (top9).pdf', dpi=300, bbox='tight')

	plt.show()

	# pairs = pairs[(x < -40) & (y < -70)]


	# for pair in pairs:

	# 	f = plt.figure()
	# 	rows, cols = 1, 2

	# 	path1 =  "BPs/" + pair[0][:4] + "/" + pair[0]
	# 	path2 =  "BPs/" + pair[1][:4] + "/" + pair[1]

	# 	ax1 = f.add_subplot(rows, cols, 1)
	# 	ax2 = f.add_subplot(rows, cols, 2)

	# 	img1 = Image.open(path1).convert('1')
	# 	img1 = np.asarray(img1)
	# 	ax1.imshow(img1, cmap='Greys_r')

	# 	img2 = Image.open(path2).convert('1')
	# 	img2 = np.asarray(img2)
	# 	ax2.imshow(img2, cmap='Greys_r')

	# 	plt.show()



def eval_model(model, env, feature_space='diff', n_eps=50, render=False, top25=False, bp_ranking=True, bp_feature_space=True):

	#observation = env.reset(init_bps=top10)
	observation = env.reset()
	# env.bp_paths = top25bps

	print(env.bp_paths)

	bp_class = []
	features = []
	pairs = []
	same_classes = []
	actions = []
	last_reward = 1
	for ep in range(n_eps):

		done = False

		while not done:

			action, _, feature = model.predict(observation, deterministic=True)

			observation, reward, done, info = env.step(action)


			current_bp = env.bp_paths[env.bp_i]
			same_class = env.current_bp_pair[-1]

			pairs.append(env.current_bp_pair)


			# print(action)

			label_map = {'a' : 0, 'b':1}

			#if render and last_reward == 0 and env.reward == 1: # or (render and env.reward == 0):
			if render and env.reward == 0 or (render and last_reward == 0):
				env.render()

			last_reward = reward

			if feature_space == 'single':
				features.append(np.array(feature[1][0].squeeze()))
				features.append(np.array(feature[1][1].squeeze()))
				bp_class += [current_bp, current_bp]

				same_classes.append(label_map[env.current_bp_pair[0][-6]])
				same_classes.append(label_map[env.current_bp_pair[1][-6]])

			elif feature_space == 'diff':
				features.append(np.array(feature[0].squeeze()))
				bp_class.append(current_bp)
				same_classes.append(same_class)
				actions.append(action)

			if done:
				observation = env.reset()

	if bp_ranking:
		plot_bp_ranking(env, top25=top25)

	if bp_feature_space:
		# plot_image_space(features, bp_class, env)
		plot_feature_space(features, bp_class, same_classes, actions, env, pairs)

	# analyze_feature_space(features, bp_class, pairs, same_classes, actions, env)

	sys.exit()

