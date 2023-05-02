# 2023/4/19
# YEAHHHH I FINIHSED GAME CLASS!!!
# This is a class where I define the DQN agent with help from
# This guy again! yeah! 
# Here is the github I referenced: https://github.com/patrickloeber/snake-ai-pytorch/blob/main/agent.py

# what libary you use, what library I use! I LOVE PLAGERIZING!
import random
import time
import numpy as np
import os.path
import torch
from collections import deque # some sort of data structure, if I can copy then why not
# the improtant ones
from WaterMazeAI import WaterMazeAI, MOVES, WIDTH
from Snakegame import SnakeGameAI, Direction, Point
from theModel import Linear_QNet, QTrainer
import matplotlib
import matplotlib.pyplot as plt
# from IPython import display


MAX_MEMORY = 100_000
BATCH_SIZE = 1500
LR = 0.001
EPOCH =  1000
EPS = 1
GAMMA = 0.99


# Now lets create my stupid kid.
class Agent:
    def __init__(self,input_layer,output_layer):
        self.num_out = output_layer
        self.n_games = 1 # avoid division by 0
        self.n_wins = 1 # avoid division by 0
        self.epsilon = EPS
        self.gamma = GAMMA
        self.memory = deque(maxlen = MAX_MEMORY) # what is this even for??
        self.model = Linear_QNet(input_layer,output_layer)
        self.trainer = QTrainer(self.model, lr=LR, gamma = GAMMA)
        self.loss_short = []
        self.loss_long = []
        
    def get_state(self,game):
        state = game.get_game_states()
        return state
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

        

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            # print("replay on sampled memory")
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        # print(rewards)
        loss = self.trainer.train_step(states, actions, rewards, next_states, dones)
        self.loss_long = loss.item()
        self.loss_short = sum(self.loss_short)/len(self.loss_short)
        
        
    def train_short_memory(self, state, action, reward, next_state, done):
        loss = self.trainer.train_step(state, action, reward, next_state, done)
        self.loss_short.append(loss.item())

    def get_action(self, state):
        # maybe i should try changing explore method
        self.epsilon = EPS - self.n_games * 0.001
        if self.epsilon < 0.1:
            self.epsilon = 0.1
        # let the model do a perdiction anyway, and overwrite it if EPS takeover
        state0 = torch.tensor(state, dtype=torch.float)
        final_move = [0] * self.num_out
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()   
        if random.random() < self.epsilon:
            move = random.randint(0, self.num_out-1)
            # print('hi',move)
        final_move[move] = 1
        return final_move

##### Some Helper Functions
def move_figure(f, x, y):
    # taken from: https://stackoverflow.com/questions/7449585/how-do-you-set-the-absolute-position-of-figure-windows-with-matplotlib
    # just a helper function to control where the figure appear on screen
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)
        
def plot_trajectory(ax,x,y):
    if len(ax.lines) >= 1:
        # if more than one trajectory exist, turn the previous one gray
        ax.lines[-1].set_color("gray")
    ax.plot(x, y, '-k', linewidth=1.0, markevery = [0], marker = '*')
    if len(ax.lines) > 20:
        # all a total of 20 line on ax
        ax.lines.pop(0)
    plt.pause(0.1)
    plt.show(block=False)
    
def plot_loss(axs,loss,reward,outcomes,n_games):
    axs[0].cla()
    axs[1].cla()
    axs[2].cla()
    x = n_games
    marker_loc = [x for x in range(len(outcomes)) if outcomes[x]]
    axs[0].set_title('Loss, long memory')
    axs[0].plot(x,loss[0],'-^b', markevery = marker_loc, linewidth = .8)
    axs[1].set_title('Loss, mean short memory')
    axs[1].plot(x,loss[1],'-*r', markevery = marker_loc, linewidth = .8)
    axs[2].set_title('Cumulative reward')
    axs[2].plot(x,reward,'-.pg', markevery = marker_loc, linewidth = .8)
    axs[0].set(ylim = (0,np.quantile(loss[0],.95)))
    axs[1].set(ylim = (0,np.quantile(loss[1],.95)))
    plt.pause(0.1)
    plt.show(block=False)
    

def game_loop_AI(game,agent):
    win = False
    while True:
        # get old state
        state_old = agent.get_state(game)
        # get move
        final_move = agent.get_action(state_old)
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


def train(agent,game,EPOCH):    
    # set up for plotting
    fig, ax = plt.subplots()
    move_figure(fig,20,100)
    fig2, ax2 = plt.subplots(3,sharex = True) # figure for loss and cum_reward eahc iteration
    move_figure(fig2,800,100)
    fig2.set_size_inches(10.2,6.4)
    ax.set(xlim=(0, game.width), ylim=(game.width,0))
    
    # pepare list to record data
    loss = [[],[]]
    cum_rewards = []
    is_win = []
    n_games = []
    
    # not really nessecary but lets just keep this here because I 
    # am scared of changing my program any more.
    plt.ioff()
    for i in range(0,EPOCH):
        # render the game play for every 100 game plays
        if agent.n_games%100 == 0:
            plot_title = f"play :{agent.n_games}win:{agent.n_wins} win rate: {agent.n_wins/agent.n_games}"
            ax.set_title(plot_title)
            print(plot_title)
            ## code to save figure, disabled for submitted version.
            # filename = f"fig{i//100}_tracePlot.png"
            # folder_path = '.\\figures'
            # filename = os.path.join(folder_path,filename)
            # fig.savefig(filename,format = 'png')
            # filename = f"fig{i//100}_loss_and_reward.png"
            # filename = os.path.join(folder_path,filename)
            # fig2.savefig(filename,format = 'png')
        # call helper function for a single game play
        win = game_loop_AI(game,agent)
        # plot trajectories
        traject = game.trajectory
        x, y = np.transpose(traject)
        
        # plot data
        plot_trajectory(ax, x, y)
        loss[0].append(agent.loss_long)
        loss[1].append(agent.loss_short)
        is_win.append(win)
        n_games.append(agent.n_games)
        cum_rewards.append(game.cum_reward)
        if len(loss[0]) == 200:
            # shrink this list when it get too large
            loss[0] = loss[0][::2]
            loss[1] = loss[1][::2]
            cum_rewards = cum_rewards[::2]
            is_win = is_win[::2]
            n_games = n_games[::2]
            
        agent.loss_short = [] # reset to store a new round of data
        plot_loss(ax2, loss, cum_rewards, is_win, n_games)
        # update game count and win counts
        agent.n_games += 1
        if win:
            agent.n_wins += 1
        game.reset()
    # save the entire model after the training session
    # torch.save(agent.model, "D:\GitHub\DS340_final_project\model\model_entire.pth")
    return agent, traject
    


def model_play(agent_trained,game):
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
            win = False
            if game.is_on_platform():
                win = True
            break
    return win
    
def validate(agent,game):
    n_wins = 0
    for i in range(0,100):
        win = model_play(agent,game)
        if win:
            n_wins += 1
    # compute total wins
    win_rate = n_wins/100
    return win_rate

# there is some problem with time updating that I need to look into.
# wired, its fine with training+rendering
if __name__ == '__main__':
    # get user input for render or not
    agent = Agent(2,2) # only allow turn or not turn
    # while True:
    #     ui_r = input("Do you wish to render with pygame, \n please type t/f (true,false): ")
    #     if ui_r == 't':
    #         render = True
    #         break
    #     elif ui_r == 'f':
    #         render = False
    #         break
    #     else:
    #         print("Not an option, please choose again.") 
    render = False
    game = WaterMazeAI(render = render)
    agent_trained, trajectory = train(agent,game,EPOCH)
    # validate performance
    new_game = WaterMazeAI(render = False)
    win_rate = validate(agent_trained,game)
    print("Validation win rate: ", win_rate)