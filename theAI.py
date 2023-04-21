# 2023/4/19
# YEAHHHH I FINIHSED GAME CLASS!!!
# This is a class where I define the DQN agent with help from
# This guy again! yeah! 
# Here is the github I referenced: https://github.com/patrickloeber/snake-ai-pytorch/blob/main/agent.py

# what libary you use, what library I use! I LOVE PLAGERIZING!
import random
import numpy as np
import torch
from collections import deque # some sort of data structure, if I can copy then why not
# the improtant ones
from WaterMazeAI import WaterMazeAI, MOVES
from theModel import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.005
EPOCH = 50
EPS = 0.9
GAMMA = 0.9


# Now lets create my stupid kid.
class Agent:
    def __init__(self):
        self.n_games = 1 # avoid division by 0
        self.n_wins = 1 # avoid division by 0
        self.epislon = EPS
        self.gamma = GAMMA
        self.memory = deque(maxlen = MAX_MEMORY) # what is this even for??
        self.model = Linear_QNet(3,5)
        self.trainer = QTrainer(self.model, lr=LR, gamma = GAMMA)
        
    def get_state(self,game):
        state = []
        state.append(game.orientation)
        state.append(game.position[0])
        state.append(game.position[1])
      #  print(state)
        return state
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = EPS - self.n_wins/self.n_games 
        # let the model do a perdiction anyway, and overwrite it if EPS takeover
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        model_move = torch.argmax(prediction).item()   
        final_move = model_move
        if random.random() < self.epsilon:
            final_move = random.randint(0, 4)

        return model_move, final_move


def train():
#    plot_scores = []
#    plot_mean_scores = []
    agent = Agent()
    game = WaterMazeAI()
    while agent.n_games < EPOCH:
        # get old state
        state_old = agent.get_state(game)

        # get move
        model_move, final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done = game.play_step(final_move)
        state_new = agent.get_state(game)
        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        # remember
        agent.remember(state_old, final_move, reward, state_new, done)
        if done:
            win = False
            if game.is_on_platform():
                print('win', str(agent.n_wins))
                agent.n_wins += 1
                win = True
            # train long memory, plot result (experience replay)
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            if win:
                # What does this step even do??
                agent.model.save()

            print('Game', agent.n_games)
        # update number of trials
    # save the entire model after while loop?
    torch.save(agent.model, "D:\GitHub\DS340_final_project\model\model_entire.pth")

def model_play():
    agent = Agent()
    agent.model = torch.load("D:\GitHub\DS340_final_project\model\model_entire.pth")
    agent.epsilon = 0
    game = WaterMazeAI()
    while True:
        # get old state
        state_old = agent.get_state(game)
        # get move
        model_move, final_move = agent.get_action(state_old)
        print(model_move,final_move)
        # perform move and get new state
        reward, done = game.play_step(final_move)
#        state_new = agent.get_state(game)
        if done:
            break

if __name__ == '__main__':
    train()
    #model_play()