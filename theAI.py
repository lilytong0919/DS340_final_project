# 2023/4/19
# YEAHHHH I FINIHSED GAME CLASS!!!
# This is a class where I define the DQN agent with help from
# This guy again! yeah! 
# Here is the github I referenced: https://github.com/patrickloeber/snake-ai-pytorch/blob/main/agent.py

# what libary you use, what library I use! I LOVE PLAGERIZING!
import random
import time
import numpy as np
import torch
from collections import deque # some sort of data structure, if I can copy then why not
# the improtant ones
from WaterMazeAI import WaterMazeAI, MOVES, WIDTH
from Snakegame import SnakeGameAI, Direction, Point
from theModel import Linear_QNet, QTrainer
import matplotlib.pyplot as plt
# from IPython import display


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.005
EPOCH = 200
EPS = 1
GAMMA = 0.9
PLOT_LINES = deque(maxlen = 20)

# Now lets create my stupid kid.
class Agent:
    def __init__(self,input_layer,output_layer):
        self.n_games = 1 # avoid division by 0
        self.n_wins = 1 # avoid division by 0
        self.epsilon = EPS
        self.gamma = GAMMA
        self.memory = deque(maxlen = MAX_MEMORY) # what is this even for??
        self.model = Linear_QNet(input_layer,output_layer)
        self.trainer = QTrainer(self.model, lr=LR, gamma = GAMMA)
        
    def get_state(self,game):
        state = game.get_game_states()
        return state
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        loss = self.trainer.train_step(states, actions, rewards, next_states, dones)
        return loss

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = EPS - self.n_wins*0.01
        # let the model do a perdiction anyway, and overwrite it if EPS takeover
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        model_move = torch.argmax(prediction).item()   
        final_move = model_move
        if random.random() < self.epsilon:
            final_move = random.randint(0, 4)

        return model_move, final_move

##### Some Helper Functions
def make_plots(ax,x,y):
    if len(ax.lines) >= 1:
        # if more than one trajectory exist, turn the previous one gray
        ax.lines[-1].set_color("gray")
    ax.plot(x, y, '-k', linewidth=1.0, markevery = [0], marker = '*')
    if len(ax.lines) > 100:
        # all a total of 20 line on ax
        ax.lines.pop(0)
    plt.pause(0.05)
    plt.show(block=False)

def game_loop_AI(game,agent):
    win = False
    while True:
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
                win = True
            break
    # experience replay after a full game
    agent.train_long_memory()
    # print(game.time_spent,game.iteration)
    # print(loss)
    return win


def train(agent,game):
#    plot_scores = []
#    plot_mean_scores = []
    #display.clear_output(wait=True)
    #display.display(plt.gcf())
    fig, ax = plt.subplots()
    ax.set(xlim=(0, game.width), ylim=(game.width,0))
    plt.ioff()
    for i in range(0,EPOCH):
        # render the game play for every 100 game plays
        if agent.n_games%100 == 0:
            print("play :", agent.n_games, " win:",agent.n_wins,
                  "win rate: ", agent.n_wins/agent.n_games)
            # TODO: code to save plots
            filename = "figures/fig_"+str(i%100)+"th_100episode"
            plt.savefig(filename,format = 'png')
        # call helper function for a single game play
        win = game_loop_AI(game,agent)
        # update game count and win counts
        agent.n_games += 1
        if win:
            agent.n_wins += 1
        # plot trajectories
        traject = game.trajectory
        x, y = np.transpose(traject)
        make_plots(ax, x, y)
        game.reset()
    # save the entire model after the training session
    torch.save(agent.model, "D:\GitHub\DS340_final_project\model\model_entire.pth")
    return agent, traject
    


def model_play(agent_trained,game):
    game.render = True
    while True:
        #print(agent.epsilon)
        # get old state
        state_old = agent_trained.get_state(game)
        # get move
        model_move, final_move = agent_trained.get_action(state_old)
#        print(model_move,final_move)
        # perform move and get new state
        reward, done = game.play_step(final_move)
#        state_new = agent.get_state(game)
        if done:
            break
        
        

# there is some problem with time updating that I need to look into.
# wired, its fine with training+rendering
if __name__ == '__main__':
    agent = Agent(6,5)
    game = WaterMazeAI()
    agent_trained, trajectory = train(agent,game)
    #game.reset()
    #model_play(agent_trained,game)