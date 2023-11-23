import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
# Import Frame Stacker Wrapper and GrayScaling Wrapper
# from gym.wrappers import GrayScaleObservation
# Import Vectorization Wrappers
# from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from collections import deque
import numpy as np


# Setup game
# env = gym_super_mario_bros.make('SuperMarioBros-v0') Wrong
# env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
# env = JoypadSpace(env, SIMPLE_MOVEMENT)

# What we are getting back from the game environment
# env.observation_space.shape  # Returns (240, 256, 3), so 240x256 RGB image (so 3-dimensional)

# What is the number of moves (buttons combinations for a single move)
# env.action_space # Returns 7, because we run the SIMPLE_MOVEMENT controls
# [['NOOP'], ['right'], ['right', 'A'], ['right', 'B'], ['right', 'A', 'B'], ['A'], ['left']]

# Play the game to get the picture
# done = True
# for step in range(100000):
#     if done:
#         env.reset()
#     # Do some random actions
#     state, reward, done, unknown, info = env.step(env.action_space.sample())
#     # Show the game on the screen
#     env.render()
# env.close()

# FrameStacker will stack 4 frames together to be able to know the direction Mario and other objects are moving
# GrayScaleObservation will convert the input RGB image (240x256x3) to Grayscale (240x256)
# stable_baseline3 is a library that we install with pip, that gives us access to several RL algorithms
# SIMPLE_MOVEMENT
# MOVES = [['NOOP'], ['right'], ['right', 'A'], ['right', 'B'], ['right', 'A', 'B'], ['A'], ['left']]
# RIGHT_ONLY
MOVES = [['NOOP'], ['right'], ['right', 'A'], ['right', 'B'], ['right', 'A', 'B']]


class Game:

    def __init__(self):
        self.states = None
        self.score = None
        self.env = None
        self.reward = None
        self.x_pos = None
        self.time = 400

        # Create the base environment
        self.env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
        # Simplify the controls
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        self.reset()

    def reset(self):
        self.score = 0
        self.reward = 0
        self.x_pos = 0

        # We will stack 4 frames in this queue, so that our game has memory. DummyVecEnv and VecFrameStack don't work
        # Images, each one consists of 3 (RGB) 240x256 frames
        self.states = deque([np.zeros((240, 256, 3), dtype=np.uint8) for _ in range(4)], maxlen=4)
        self.env.reset()
        self.env.render()
        # Grayscale. keep_dim=True is required for stacking our frames later. Our input becomes 240x256x1
        # I disabled it because alexnet is usually trained on RGB images, and we use a pretrained network
        # self.env = GrayScaleObservation(self.env, keep_dim=True)
        # Wrap inside the Dummy Environment. Our input becomes1x240x256x1
        # self.env = DummyVecEnv([lambda: self.env])
        # Stack the frames, so that our model has 'memory'
        # self.env = VecFrameStack(self.env, 4, channels_order='last')  # Stack 4 frames. Our input becomes 1x240x256x4

    def play_step(self, action):
        state, reward, done, unknown, info = self.env.step(action)
        self.calculate_reward(info, done)
        self.states.append(state)
        return self.reward, done, self.score

    def calculate_reward(self, info, done):
        # The reward function assumes the objective of the game is to move as far right as possible
        # (increase the agent's x value), as fast as possible, without dying.
        # Not examining if it reached the flag, since going right it will eventually reach it without dying hopefully
        x_pos = info['x_pos']
        time = info['time']

        # Motivate Mario to move right and reach the flag
        self.reward += x_pos - self.x_pos
        self.reward += time - self.time

        # Motivate Mario to avoid dying
        if done:
            self.reward -= 15

        self.x_pos = x_pos
        self.time = time
