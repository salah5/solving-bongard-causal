import gym
import random
import os
import sys
import numpy as np
from gym import spaces
from PIL import Image
import itertools
import matplotlib.pyplot as plt

from pprint import pprint

class BPEnv(gym.Env):

  def __init__(self, ep_length=None, no_state=False, label_in_state=False, flat_state=False, small_state=False, unbalanced=False):
    super(BPEnv, self).__init__()

    self.label_in_state = label_in_state
    self.no_state = no_state
    self.flat_state = flat_state
    self.small_state = small_state
    self.unbalanced = unbalanced
    self.ep_length = ep_length

    self.observation_space = spaces.Box(low=0, high=1, shape=
                    (2, 97, 97), dtype=np.int64)
    if self.flat_state:
      self.observation_space = spaces.Box(low=0, high=1, shape=(18818,), dtype=np.int)
    if self.small_state:
      self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.int)
    self.action_space = spaces.Discrete(2)

    self.bp_queue = []

  def step(self, action):

    next_bp = self.bp_queue.pop(0)
    next_state = self._bp_pair_to_state(next_bp)
    done = not self.bp_queue

    reward = 1 if action == self.current_bp_pair[-1] else 0

    # print(self.current_bp_pair)
    # print(f'{self.current_bp_pair=}, {action=}, {reward=}')

    self.current_bp_pair = next_bp
    
    return next_state, reward, done, {}


  def reset(self):
    
    bp_paths = os.listdir("BPs")
    bp_paths = list(filter(lambda x: not x.startswith("."), bp_paths))

    init_bp = random.choice(bp_paths)
    self.bp_path = "BPs/" + init_bp + "/"

    bp_dirs = os.listdir(self.bp_path)
    bp_dirs = list(filter(lambda x: len(x) > 8 and (not x.startswith(".")), bp_dirs))

    # bp_pairs1 = list(itertools.combinations(bp_dirs, 2))
    bp_pairs = list(itertools.product(bp_dirs, bp_dirs))

    if self.ep_length:
      bp_pairs = random.sample(bp_pairs, self.ep_length)

    self.bp_queue = []
    k = 0
    for bp_pair in bp_pairs:

      k += 1

      if bp_pair[0][-6] == bp_pair[1][-6]:
        if self.unbalanced and (k % 3 == 0):
          continue
        self.bp_queue.append(list(bp_pair) + [1])
      else:
        self.bp_queue.append(list(bp_pair) + [0])


    random.shuffle(self.bp_queue)


    init_bp_pair = self.bp_queue.pop(0)
    self.current_bp_pair = init_bp_pair

    state = self._bp_pair_to_state(self.current_bp_pair)

    return state

    
  def render(self, mode='human', close=False):

    imgs = self._bp_pair_to_state(self.current_bp_pair)

    f = plt.figure()  
    f.add_subplot(1,2, 1)
    plt.imshow(imgs[0])
    f.add_subplot(1,2, 2)
    plt.imshow(imgs[1])
    plt.show(block=True)

  def _bp_pair_to_state(self, bp_pair):
    # print(f'{bp_pair=}')

    image1 = Image.open(self.bp_path + bp_pair[0])
    image2 = Image.open(self.bp_path + bp_pair[1])

    image1 = image1.convert('1')
    image2 = image2.convert('1')
    # sys.exit()
    image1 = np.asarray(image1)
    image2 = np.asarray(image2)


    state = np.array([image1, image2]).astype(int)

    if self.label_in_state:

      names = [bp_pair[0], bp_pair[1]]

      for i in range(2):

        if names[i][-6] == 'a':
          state[i].fill(0)
        if names[i][-6] == 'b':
          state[i].fill(1)


      if self.small_state:
        if names[0][-6] == names[1][-6]:
          return np.ones(5)
        else:
          return np.zeros(5)

    if self.flat_state:
      state = np.concatenate((state[0].squeeze(), state[1].squeeze())).flatten()

    if self.no_state:
      state.fill(0)

    

    return state

