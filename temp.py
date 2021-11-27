import gym 
import numpy as np
import time
# from agent import Agent
import cv2
from PIL import ImageGrab
import matplotlib.pyplot as plt
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
def screen_record(): 
    last_time = time.time()
    printscreen =  np.array(ImageGrab.grab(bbox=(0,40,1920,1080)))
    print('loop took {} seconds , shape = {}'.format(time.time()-last_time, printscreen.shape))
    last_time = time.time()
    cv2.imwrite('scr.png', printscreen)
    return printscreen


env = gym.make('my_env:foraging-v0')
print(env.observation_space.shape)
done = False
score = 0
x = []
start_time = time.time()
while not done:
    img, next_state, tim= env.render()
    corr, juice, done, info = env.step(np.random.randint(0,8))
    score += juice
    x.append(score)


y = []
for i in range(len(x)):
    y.append(i)

plt.plot(y, x)
plt.show()
directions = []
