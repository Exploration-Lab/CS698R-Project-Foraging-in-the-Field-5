
import gym
from gym import spaces
from gym.utils import seeding
from my_env.envs.garden import Garden
import xlrd
import numpy as np
import cv2
import pandas as pd 
import random
mode = 'agent'

class foraging(gym.Env):
  def __init__(self):
    self.pygame = Garden()
    self.berx_cor, self.bery_cor, self.sz = self.get_pos(0)
    # self.seed()
    self.x_cor = 800
    self.y_cor = 500
    self.juice = 0.5
    self.dist = 0
    self.collec = 0
    self.time_rem = 120
    self.action_space = spaces.Discrete(8)
    self.observation_space = spaces.Box(low=0, high=255, shape=(1920, 1080, 3))
    # self.observation_space = spaces.Discrete(2)
  # def seed(self, seed = None):
  #   self.np_random, seed = seeding.np_random(seed)
  #   return [seed]


  def get_pos(self, t):
    if t != 0:
      loc = ('my_env/envs/berries_loc'+str(t)+'.xlsx')
    else:
      loc = ('my_env/envs/berries_loc.xlsx')
    data = pd.read_excel(loc)
    berx_cor = data['x'].to_numpy().tolist()
    bery_cor = data['y'].to_numpy().tolist()
    sz = data['size'].to_numpy().tolist()
    for i in range(len(bery_cor)):
      berx_cor[i] -= 10000
      bery_cor[i] -= 11000
    # print(min(berx_cor), min(bery_cor), max(berx_cor), max(bery_cor))
    return berx_cor, bery_cor, sz

  def check_collision(self):
    d = []
    self.collec = 0
    for i in range(len(self.berx_cor)):
      dist = np.sqrt((self.x_cor - self.berx_cor[i])**2 + (self.y_cor - self.bery_cor[i])**2)
      if dist <= self.sz[i] + np.random.normal(10, 3):
        self.juice += self.sz[i]/1000
        self.collec += self.sz[i]/1000
        d.append(i)
    for i in d:
      if i < len(self.berx_cor):
        self.berx_cor.pop(i)
        self.bery_cor.pop(i)
        self.sz.pop(i)

#Returns [Reward, Regret, Optimal Action]
  def step(self, action):
    self.check_collision()
    prev_x, prev_y = self.x_cor, self.y_cor
    self.x_cor, self.y_cor, done, self.time_rem = self.pygame.action(action, mode)
    self.dist = np.sqrt((prev_x - self.x_cor)**2 + (prev_y - self.y_cor)**2)
    self.juice -= 0.5*self.dist/(20*560)
    q = -0.5*self.dist/(20*560)
    return [self.x_cor, self.y_cor], self.collec*100  , done, self.juice

  def render(self):
    w = self.pygame.view(self.berx_cor, self.bery_cor, self.sz, self.juice)
    return w

  def create_file( self, seed = 100, n_patch = 10, env = (20000 , 20000), patch = (1500 , 1500), path = './berry_loc.json' , cnt = 0):
    """
    path :- File address to save berry locations.
    """
    berries = [(10,20),(20,20),(30,20),(40,20)]		# (size , count)

    # rng = random.Random(seed)

    centx = [10989.56234,10061.28228
          ,4277.277945
          ,5983.81127
          ,15424.18285
          ,4185.375819
          ,18191.01981
          ,10517.87811
          ,17072.07713
          ,1395.563327]

    centy = [5503.125429
          ,11138.30319
          ,11423.9542
          ,4684.407793
          ,13970.4661
          ,16599.27841
          ,9608.029913
          ,18148.06928
          ,1802.178798
          ,7256.295653
          ]
    centx = [int(a) for a in centx]
    centy = [int(a) for a in centy]
    cents = list(zip(centx , centy))
    x = []
    y = []
    sz = []
    p = []
    for c in cents:
      for s , n in berries:
        bx = random.sample(range(c[0] - patch[0] , c[0] + patch[0]) , k=n)
        by = random.sample(range(c[1] - patch[1] , c[1] + patch[1]) , k=n)
        for i in range(len(bx)):
          x.append(bx[i])
          y.append(by[i])
          sz.append(s)
          p.append(0)
    df = pd.DataFrame(list(zip(p, sz, x, y)), columns =['patch', 'size', 'x', 'y'])
    df.to_excel('my_env/envs/berries_loc'+str(cnt)+'.xlsx')

  def reset(self, t):
    self.x_cor = 800
    self.y_cor = 500
    t = 0
    if t != 0:
      self.create_file(cnt = t)
    self.berx_cor, self.bery_cor, self.sz = self.get_pos(t)
    self.juice = 0.5
    self.dist = 0
    self.collec = 0
    self.time_rem = 120
    self.pygame = Garden()
    return np.zeros(34)