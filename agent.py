from game import Game, MOVES
from model import Linear_QNet
import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.0001


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # If mem is exceeded, it will popleft()
        self.model = Linear_QNet(64, 256, len(MOVES))  # len(MOVES) to switch between modes easily
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        # Convert the deque data into a single NumPy array
        states = np.array(game.states, dtype=int)

        # Normalize pixel values (if needed) and transpose dimensions
        states = states.astype(np.float32) / 255.0  # Normalize pixel values between 0 and 1
        states = np.transpose(states, (0, 3, 1, 2))  # Adjust dimensions to match PyTorch format

        return states

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Random moves: tradeoff between exploration and exploitation
        self.epsilon = 80 - self.n_games

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, len(MOVES) - 1)  # len(MOVES) to switch between modes easily
        else:
            state0 = (torch.tensor(state, dtype=torch.float)).unsqueeze(0)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

        return move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = Game()

    while True:
        # Get old state
        state_old = agent.get_state(game)

        # Get move
        final_move = agent.get_action(state_old)

        # Perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            plot_scores.append(score)
            total_score += score
            mean_score = total_score/agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
