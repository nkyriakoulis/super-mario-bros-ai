# Super Mario Bros with Reinforcement Learning

An AI that learns to play Super Mario Bros. 
Mario environment: https://github.com/Kautenja/gym-super-mario-bros

## Model
The model is a hybrid NN. The first part is a pretrained AlexNet CNN that takes 
the last 4 frames of the game as input (240x256 RGB) and outputs a 4x16 tensor.
The tensor is flattened and then fed to a Q Network of 3 layers 
(64, 256, number_of_moves units).
We can choose between three possible sets of moves from:
https://github.com/Kautenja/gym-super-mario-bros/blob/master/gym_super_mario_bros/actions.py

## Reward Function
The reward function assumes the objective of the game is to move as far right 
as possible (increase the agent's x value), as fast as possible, without dying.
Not examining if it reached the flag at the end of the stage, since, going right,
it will eventually reach it without dying, hopefully.

## Setup environment
The project was set up in an anaconda env. Make sure to install all the
dependencies in order to run it.

```
pip install gym-super-mario-bros
pip install torch torchvision
pip install ipython
```