# CS698R-Project-Foraging-in-the-Field-5

Project by Group-5 DarkBox. Contains implementation of environment and agent.

## Environment
### Dependency
- OpenGym
- PyGame
### Instructions
Install gym environment 
```console
$ pip install -e .
```

## Approch to solution of the problem
The Double Deep-Q Learning (DDQN) algorithm is used to train the agent. The state size of the agent is 35, and the action size is either 8 or 9 depending on whether it has the option to stay. <br>
The features can be divided into 4 vectors of length 8. Each vectors denotes the direction: [N, S, E, W, NE, NW, SE, SW]. The agent analyzes  each of the contained direction and if it encounters a berry of size say ’n’ it fills the corresponding indices with  size/distance   for the vector corresponding to size ’n’. Therefore we have 8*4 = 32 states.<br>

The 33rd  state is density which tells how much of the visible screen is occupied by the berries.  The 34th and 35th states are the remaining health and remaining time respectively. The features are scaled to give the best results. <br>

Appropriate noise is added to the states to prevent over-fitting.

## Agent
### Dependency
- PyTorch

### Instructions
To train agent, use command
```console
$ python run.py
```
