
from agent import Agent
from collections import deque
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
from torchvision import models
from torchsummary import summary

env = gym.make('my_env:foraging-v0')
agent = Agent(state_size=34,action_size=8,seed=0)
agent.qnetwork_local.load_state_dict(torch.load('checkpoint345'))
agent.qnetwork_target.load_state_dict(torch.load('checkpoint345target'))
# print(agent.qnetwork_target.parameters)
# for name, param in agent.qnetwork_target.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)
def dqn(n_episodes= 70, max_t = 50000, eps_start=1, eps_end = 0,
       eps_decay=0.996):

    scores = [] # list containing score from each episode
    scores_window = deque(maxlen=100) # last 100 scores
    eps = eps_start
    for i_episode in range(1, n_episodes+1):
        x = []
        y = []
        state = env.reset(0)
        score = 0
        # eps = 1
        prev_juice = 0.5
        action = 0
        for t in range(max_t):
            img, next_state, tim = env.render()
            if np.random.random() < 0.5:
            # if t%3 == 0:
                action = agent.act(state,0)
            coor,reward,done,juice = env.step(action)
            next_state = list(next_state)
            next_state.append(juice)
            # next_state.append(action)
            next_state = np.array(next_state)
            agent.step(state,action,reward,next_state,done)
            state = next_state
            score += (juice - prev_juice)
            score += reward
            prev_juice = juice
            if done:
                # score += tim
                break
            coor.append(score)
            coor.append(juice)
            coor.append(tim)
            coor.append(action)
            x.append(coor)
            scores_window.append(score) ## save the most recent score
            scores.append(score) ## sae the most recent score
        print('\rEpisode {}\tAverage Score {:.2f} \t eps = {}'.format(i_episode,np.mean(scores_window), eps))
        
        file_name = 'agent_path_test_a' + str(i_episode)
        with open(file_name, 'w') as f:
            csv.writer(f, delimiter=' ').writerows(x)
    return scores

scores = dqn()
# print(scores)
# with open('scores.txt', 'w') as f:
#     csv.writer(f, delimiter=' ').writerows(scores)
