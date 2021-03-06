# CS698R-Project-Foraging-in-the-Field-5

Project by Group-5 DarkBox. Contains implementation of environment and agent.

## Abstract
Foraging behaviour is a highly studied topic in fields like neuro science and computer science. From neuro science perspective,  foraging theory deals with observing and explaining foraging behaviours of species in response to the environment it lives in.  This problem helps in studying decision making of species to continue their journey in the hope of finding new patch of resources given the cost of travelling.
From reinforcement learning perspective, it is an optimization problem. The objective of the project is to present an agent which can forage optimally in the given environment. By using Deep Reinforcement Learning, we try to understand how to forage optimally in a field with patches of berries.

## Approach to solution of the problem
The Double Deep-Q Learning (DDQN) algorithm is used to train the agent. The state size of the agent is 35, and the action size is either 8 or 9 depending on whether it has the option to stay. <br>
The features can be divided into 4 vectors of length 8. Each vectors denotes the direction: [N, S, E, W, NE, NW, SE, SW]. The agent analyzes  each of the contained direction and if it encounters a berry of size say ānā it fills the corresponding indices with  size/distance   for the vector corresponding to size ānā. Therefore we have 8*4 = 32 states.<br>

The 33rd  state is density which tells how much of the visible screen is occupied by the berries.  The 34th and 35th states are the remaining health and remaining time respectively. The features are scaled to give the best results. <br>

Appropriate noise is added to the states to prevent over-fitting.

## Environment
OpenAI environment interface can be found in "my_env\envs\foraging.py". The OpenAI gym interface is integrated with the Pygame to render the game. The "foraging.py" files also provides the "mode" option which can be changed to "human" if someone wants to test the game out. Make sure to change the mode back to agent before the training. The step function of the environment takes the action and returns [current coordinates, current berry collected, done, current health]. The render function is capable of returning the current image, next_state and the time remaining. To return the image uncomment the code from line 217-222 in "garden.py" (it is currently commented because it slows down the game drastically).

### Dependency
- OpenGym
- PyGame
### Instructions
Install gym environment 
```console
$ pip install -e .
```

## Agent
### Dependency
- PyTorch

### Instructions
To see the environment use:
```console
$ python temp.py
```

To train agent, use command
```console
$ python run.py
```
To run trained agent,
```console
$ python test.py
```
The agent's checkpoints will be saved after every 10 episodes or if the score reaches greater than 200 along with the agents path, score vs time, health vs time. 
Provide the agent's path file and use:
```console
$ python plotting.py
```
to see the results of agent's performance on various parameters.

The environment also supports random environment. To train on random environments, in line '43' of 'run.py' instead of passing 0 to env.reset pass the current episode number to create new environment with each episode.
### References:
1) CS698R Reinforcement Learning course IITK by Prof. Ashutosh Modi
2) Udacity Reinforcement Learning course

[//]: [CS698R-Project-Presentation-5.pdf](https://github.com/Exploration-Lab/CS698R-Project-Foraging-in-the-Field-5/files/7613752/CS698R-Project-Presentation-5.pdf)
