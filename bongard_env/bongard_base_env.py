import gym
import random
import os
import sys
import numpy as np
from gym import spaces
from PIL import Image
import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pprint import pprint

class BPEnv(gym.Env):

  def __init__(self, eval=False, ep_length=100, skip_action=True):
    super(BPEnv, self).__init__()

    self.ep_length = ep_length
    self.skip_action = skip_action

    self.observation_space = spaces.Box(low=0, high=1, shape=
                    (2, 97, 97), dtype=np.int64)


    self.action_space = spaces.Discrete(3) if self.skip_action else spaces.Discrete(2)

    bp_paths = os.listdir("BPs")
    self.bp_paths = list(filter(lambda x: not x.startswith("."), bp_paths))
    
    # self.bp_paths = self.bp_paths[:75] if not eval else self.bp_paths[74:]

    self.bp_i = 0
    self.bp_success = {}
    self.bp_attention = {}

  def step(self, action):

    done = False
    info = {}
    self.prev_pair = self.current_bp_pair 
    action += 1 if not self.skip_action else 0

    if action == 0 or self.current_bp_pair[0] == self.current_bp_pair[1]:
      self.reward = 0
      new_example = random.sample(self.bp_dirs, 1)[0]
      bp_pair = [self.current_bp_pair[0], new_example]


    else:
      self.reward = 1 if action == self.current_bp_pair[-1] else -1
      bp_pair = random.sample(self.bp_dirs, 2)

    self.bp_success[self.bp] += self.reward

    # done = True if self.ep_step > self.ep_length - 2 or self.reward == -1 else False
    done = True if self.ep_step > self.ep_length - 2 else False

    state, label = self._bp_pair_to_state(bp_pair)
    self.current_bp_pair = bp_pair + [label]

    self.ep_step += 1

    return state, self.reward, done, info

  def reset(self, init_bps=[]):

    if init_bps:
      self.bp_paths = init_bps

    self.bp = self.bp_paths[self.bp_i]
    self.bp_path = "BPs/" + self.bp + "/"

    if self.bp not in self.bp_success:
      self.bp_success[self.bp] = 0

    bp_dirs = os.listdir(self.bp_path)
    self.bp_dirs = list(filter(lambda x: len(x) > 8 and (not x.startswith(".")), bp_dirs))
    
    bp_pair = list(random.sample(self.bp_dirs, 2))

    state, label = self._bp_pair_to_state(bp_pair)

    self.current_bp_pair = bp_pair + [label]

    self.ep_step = 0
    self.reward = 0
    self.prev_pair = self.current_bp_pair
    self.bp_i = self.bp_i + 1 if self.bp_i < len(self.bp_paths) - 1 else 0

    return state

  def render2(self, mode='human', close=False):

    imgs, _ = self._bp_pair_to_state(self.current_bp_pair[:-1])

    f = plt.figure()  
    f.add_subplot(1,2, 1)
    plt.imshow(imgs[0])
    ax = f.add_subplot(1,2, 2)
    ax.annotate(str(self.prev_reward), (80,10))
    plt.imshow(imgs[1])

    plt.show(block=True)

  def render(self, mode='human', close=False):

    bp_image = self.bp_path + self.prev_pair[0][:4] + '.gif'

    bp_image = Image.open(bp_image)
    bp_image = 1 - np.asarray(bp_image.convert('1'))


    correct = 'red' if self.reward < 1 else 'lime'

    selected_areas = self.prev_pair[0][-6:-4], self.prev_pair[1][-6:-4]

    bp_areas = {'a0' : (5, 5), 'a1' : (115, 5), 'a2' : (5, 114), 'a3' : (115, 114), 'a4' : (5, 222), 'a5' : (115, 222),
                'b0' : (300, 5), 'b1' : (409, 5), 'b2' : (300, 114), 'b3' : (409, 114), 'b4' : (300, 222), 'b5' : (409, 222)}
    
    plt.figure()
    currentAxis = plt.gca()
    currentAxis.imshow(bp_image, cmap='Greys')
    currentAxis.add_patch(Rectangle(bp_areas[selected_areas[0]],  100, 100, linewidth=2, color=correct, fill=None, alpha=1))
    currentAxis.add_patch(Rectangle(bp_areas[selected_areas[1]],  100, 100, linewidth=2, color=correct, fill=None, alpha=1))
    currentAxis.annotate('Reward: ' + str(self.reward), (200, 0))
    plt.show(block=True)

  def _bp_pair_to_state(self, bp_pair):

    image1 = Image.open(self.bp_path + bp_pair[0])
    image2 = Image.open(self.bp_path + bp_pair[1])

    image1 = image1.convert('1')
    image2 = image2.convert('1')
    # sys.exit()
    image1 = np.asarray(image1)
    image2 = np.asarray(image2)

    label = 1 if bp_pair[0][-6] == bp_pair[1][-6] else 2
    state = np.array([image1, image2]).astype(int)

    return state, label

