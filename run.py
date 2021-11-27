
from agent import Agent
from collections import deque
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import seaborn as sns; sns.set()
# from sklearn.cluster import KMeans
os.environ['KMP_DUPLICATE_LIB_OK']='True'


env = gym.make('my_env:foraging-v0')
agent = Agent(state_size=34,action_size=8,seed=0)

# agent.qnetwork_local.load_state_dict(torch.load('checkpoint55'))
# agent.qnetwork_target.load_state_dict(torch.load('checkpoint55target'))
# from torchsummary import summary
# summary(agent.qnetwork_local, (1, 35, 35))



def ddqn(n_episodes= 500, max_t = 50000, eps_start=1.0, eps_end = 0.01,
       eps_decay=0.996):

    scores = [] # list containing score from each episode
    scores_window = deque(maxlen=100) # last 100 scores\
    plt.ion()
    fig, ax = plt.subplots()
    eps = eps_start
    reward_list = []
    action = 0
    tim = 0
    berr_x = []
    berr_y = []
    berr = []
    score = 0
    for i_episode in range(1, n_episodes+1):
        x = []
        y = []
        state = env.reset(0)
        # eps = 1
        prev_juice = 0.5
        curr_d = 0
        prev_x = 800
        prev_y = 500
        score2 = 0
        for t in range(max_t):
            img, next_state, tim = env.render()
            if np.random.random() < 0.5:
                action = agent.act(state,eps)
            coor,reward,done,juice = env.step(action)
            next_state = list(next_state)
            next_state.append(juice)
            # next_state.append(tim/1000)
            next_state = np.array(next_state)
            agent.step(state,action,reward,next_state,done)
            state = next_state
            score += (juice - prev_juice)
            score += reward
            score2 += reward
            score2 += (juice - prev_juice)
            prev_juice = juice
            
            if done:
                # score += tim
                break
            coor.append(score2)
            coor.append(juice)
            coor.append(tim)
            coor.append(action)
            # coor.append(action)

            x.append(coor)
            scores_window.append(score2) ## save the most recent score
            scores.append(score) ## sae the most recent score
            eps = max(eps*eps_decay,eps_end) ## decrease the epsilon
            print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,score2), end = "")


        if i_episode %10==0 or score > 250:
            print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(scores_window)))
            name = 'checkpoint'+str(i_episode)
            torch.save(agent.qnetwork_local.state_dict(),name)
            torch.save(agent.qnetwork_target.state_dict(), name+"target")
            file_name = 'paths/agent_path'+str(i_episode)
            with open(file_name, 'w') as f:
                csv.writer(f, delimiter=' ').writerows(x)


        reward_list.append(score/i_episode)
        ax.set_xlim(0,300)
        ax.cla()
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Average Reward')
        ax.plot(reward_list, 'g-', label='total_loss')
        plt.pause(0.001)
        

    return scores

scores = ddqn()
print(scores)
with open('scores.txt', 'w') as f:
    csv.writer(f, delimiter=' ').writerows(scores)