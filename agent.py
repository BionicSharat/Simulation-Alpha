import torch
import random
from sim import SimulationAI
from collections import deque
import numpy as np
from model import QTrainer, Linear_QNet
from collections import Iterable
import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

def flatten(items):
    for elem in items:
        if isinstance(elem,Iterable) and not isinstance(elem,str):
            for sub_elem in flatten(elem):
                yield sub_elem
        else:
            yield elem

class Agent:

    def __init__(self):
        self.n_games = 0
        self.n_runs = 0
        self.model = Linear_QNet(48, 512, 28)
        self.gamma = 0.99 # discount rate
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.epsilon = 0 # random rate
        self.memory = deque(maxlen=MAX_MEMORY)

    def get_state(self, game):
        state = [[list(i.values()) for i in game.iceBergs]]
        # print(state)
        state = list(flatten(state))
        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # pop left when max memorry reach
    
    def train_long_memory(self):
        if (len(self.memory) < BATCH_SIZE):
            mini_sample = random.sample(self.memory, BATCH_SIZE if len(self.memory) > BATCH_SIZE else len(self.memory)) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        for state, action, reward, next_state, done in mini_sample:
             self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 50 - self.n_runs # action => [ [[startId, endId, troopsNum] * x] , [upgradeId] ]
        final_move = []
        if random.randint(0, 80) < self.epsilon:
            for i in range(7):
                final_move.extend([random.randint(0, 7), random.randint(0,7), random.randint(0, 100)])
            for i in range(7):
                final_move.append(random.randint(0, 1))
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move_tensor = prediction.flatten()
            move_tensor = move_tensor.to(torch.int64)
            move = move_tensor.tolist()
            final_move = move
        
        return final_move
    
def train():
    avg = []
    n_games_l = []
    avg_sum = 0
    shortest_game = 1e100
    agent = Agent()
    sim = SimulationAI(10000, 10000)
    while True:
        # get old state
        state_old = agent.get_state(sim)
        # get action
        final_action = agent.get_action(state_old)
        # play and get new state
        reward, win, turn = sim.play(final_action)
        state_new = agent.get_state(sim)
        # train short memory
        agent.train_short_memory(state_old, final_action, reward, state_new, win)
        # remember
        agent.remember(state_old, final_action, reward, state_new, win)

        if win:
            # train long memoery
            sim.reset()
            agent.n_games += 1
            avg_sum += turn
            avg.append(avg_sum/agent.n_games)
            n_games_l.append(agent.n_games)
            agent.train_long_memory()
            if turn < shortest_game:
                shortest_game = turn
                agent.model.save()
            # print("avg", avg_sum/agent.n_games)
            
            # print('Game Number: {0} | Turns: {1} | Fastest game: {2}'.format(agent.n_games, turn, shortest_game))

            plot.plot(n_games_l, avg)
if __name__ == '__main__':
    train()