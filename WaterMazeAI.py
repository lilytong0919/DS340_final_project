# 2023/4/18
# The AI adapted WaterMaze class (although the non-adapted version don't exist)
# guidance for class strcutre: https://www.youtube.com/watch?v=L8ypSXwyBds
# github page for above resource: https://github.com/patrickloeber/snake-ai-pytorch
# startup code with chatGPT, see appendex.

import pygame
import numpy as np
import math
import random
from enum import Enum
import copy


# a Action class that stores all possible actions
class MOVES:
    CLOCKWISE = 0
    COUNTER_CLOCKWISE = 1
    FASTER = 2
    SLOWER = 3


# Set RGB Color needed
WHITE = (236,240,236)
RED = (255, 0, 0)
BLACK = (0,0,0) # potentially plot a trajectory
GRAY = (150,160,150)
BLUE = (0,100,128)

# unchanging part of the game
FPS = 30
MAX_SPEED = 500
POOL_RADIUS = 300
WIDTH = POOL_RADIUS*2 + 50
POOL_CENTER = [WIDTH/2, WIDTH/2]
PLATFORM = [POOL_RADIUS/2, POOL_RADIUS/2]
PLATFORM_RADIUS = POOL_RADIUS/5

START = [POOL_CENTER[0], POOL_RADIUS*2-20]
ORIENT = random.randint(0,360)

# limit game length
MAX_EPISODE_TIME = 30 # time in second, simulated time of each trial
EN_SCALE = 0.005 # larger then energy spend faster
MS_ITERATION = 33 # time elapse per iteration if no rendering, match to 30 fps

# the game class
class WaterMazeAI:
    def __init__(self, w = WIDTH, h = WIDTH, render = False):
        self.width = w
        self.height = h
        self.render = render
        # set up the display window?
        if self.render:
            pygame.init()
            self.display = pygame.display.set_mode((self.width,self.height))
            self.font = pygame.font.SysFont('arial', 20)
            pygame.display.set_caption('Water Maze')
        # not sure what clock do, but lets set it first
            self.clock = pygame.time.Clock()
        # platform location is fixed so I will set it here.
        self.platform = PLATFORM # the center point of plat form
        self.reset()
        
    def reset(self):
        # upon reset the game, we should place the 'mouse' at a random location
        # facing a random direction
        self.speed = 50
        self.position = START
        self.iteration = 0
        self.trajectory = [] # *math.ceil(MS_ITERATION*1000/MAX_EPISODE_TIME) # keep track of past positions, for ploting
        self.cum_reward = 0
        
        # code to set random start each trial, disabled for submission
        # while not self.is_in_circle(POOL_CENTER,POOL_RADIUS,self.position) \
        #       and not self.is_on_platform():
        #     self.position = [random.randint(0,WIDTH),
        #                       random.randint(0,WIDTH)]
            
        self.orientation = random.randint(0,360) # ORIENT # orientation in degree
        self.energy = 100
        self.time_spent = 0
        # consider adding a visibility range
    
    def is_in_circle(self,center,radius,position):
        result = False
        dist2center = np.linalg.norm(np.array(center) - np.array(position))
        if dist2center < radius:
            result = True
        return result
    
    def is_on_platform(self):
        # check if mice is on plat form
        return self.is_in_circle(PLATFORM,PLATFORM_RADIUS,self.position)
        
    def is_game_over(self):
        result = False
        if self.is_on_platform():
            result = True
        elif self.energy <= 0:
            result = True
        elif self.time_spent >= MAX_EPISODE_TIME: # Each trial 1 min max
            result = True
        return result
        
    def play_step(self,action):
        reward = 0
        game_over = False
        # 1 collect input I don't know what this is for, but let just put it in and see
        # the training seems to need action as an bool array, here I change it back to number
        action = np.argmax(action)
        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
        
        # prev_position = np.array(self.position)
        # 2 move
        
        #print('hihi')
        self.move(action)    
        # I hate shallow copy! its always the shallow copy that let me sit 
        # there debug for 2 hour without figuring out why!
        self.trajectory.append(copy.deepcopy(self.position))
        # 3 check if game is over
        if self.is_game_over():
            # reward = -5 # punish if not on platform at the end
            if self.is_on_platform():
                reward = 100 # + self.energy + (MAX_EPISODE_TIME - self.time_spent) # Reward saving energy and reaching goal fast
            game_over = True           
        else:
            if self.get_angle_diff() < 15:
                reward = 1
            else:
                reward = -1
            # add a reward for actively moving and see what happens
            # expected movement - actually moved.
            # reward = np.linalg.norm((self.trajectory[-1]-np.array(self.position)))
            #print(reward)
        # 4 update ui and clock
        if self.render:
            self.update_ui()
        else:
            self.iteration += 1
        # 5 store values I wanted to "record"
        # 6 return values
       #  print(reward)
        self.cum_reward += reward
        return reward, game_over
    
    def update_ui(self):
        
        # update the display?
        self.display.fill(WHITE)        
        # Platform should be invisible to the AI, but lets plot it for now
        pygame.draw.circle(self.display, BLUE, POOL_CENTER, POOL_RADIUS + 20)
        pygame.draw.circle(self.display, GRAY, PLATFORM, PLATFORM_RADIUS)
        
        # text information here:
        text_lines = [self.font.render("Energy: " + str(self.energy), True, BLACK),
                      self.font.render("Time_spent: " + str(self.time_spent), True, BLACK)]
        text_positions = [[0, 0],[0, 20]]
        for i in range(len(text_lines)):
            self.display.blit(text_lines[i], text_positions[i])
        
        # arrow should be draw last in any case.
        self.draw_arrow()
        # update the screen
        pygame.display.flip()
        

    def draw_arrow(self):
        # Convert the orientation from degrees to radians
        radian = math.radians(self.orientation)

        # Define the vertices of the arrow
        vertices = np.array([[20, 0], [15, 5], [0, 0], [15, -5]])
        # Rotate the vertices based on the orientation
        rotation_matrix = np.array([[math.cos(radian), -math.sin(radian)], 
                                    [math.sin(radian), math.cos(radian)]])

        rotated_vertices = vertices.dot(rotation_matrix) 
        # Translate the vertices based on the position
        translated_vertices = rotated_vertices + np.reshape(self.position,(1, 2))

        # Draw the arrow on the screen
        pygame.draw.polygon(self.display, RED, translated_vertices)



    def move(self,action):
        # make changes to the 'mouse' position and orientation given action
        # get index of action
        if action == MOVES.CLOCKWISE:
            self.orientation += 5
        elif action == MOVES.COUNTER_CLOCKWISE:
            self.orientation -= 5
        elif action == MOVES.FASTER:
            self.speed += 10
        elif action == MOVES.SLOWER:
            self.speed -= 10
        
        # its fine to not do this step, but I feel like its good to keep the 
        # orientation within the 0-360 range
        if self.orientation < 0 or self.orientation > 360:
            self.orientation %= 360
        # keep speed in control
        if self.speed > MAX_SPEED:
            self.speed = MAX_SPEED
        elif self.speed <= 0:
            self.speed = 0
        
        # I will put position calculation here for now.
        radian = math.radians(self.orientation)
        direction = np.array([math.cos(radian), -math.sin(radian)])
        if self.render:
            time_elapsed = self.clock.tick(FPS)
        else:
            time_elapsed = MS_ITERATION
        # update position of the 'mice'
        # I might be better off computing speed with time considered, as 
        # seems to change with frame rate.
        change_position = self.speed * direction * time_elapsed/1000
        # let the 'mouse' stay where it was if attempt to move into the wall
        # update only the component to position where movement is allowed
        move_x = copy.deepcopy(self.position)
        move_x[0] += change_position[0]
        move_y = copy.deepcopy(self.position)
        move_y[1] += change_position[1]
        if self.is_in_circle(POOL_CENTER,POOL_RADIUS, (self.position+change_position)):
            self.position += change_position
        elif self.is_in_circle(POOL_CENTER,POOL_RADIUS, move_x):
            self.position = move_x
        elif self.is_in_circle(POOL_CENTER,POOL_RADIUS, move_x):
            self.position = move_y
        self.energy -= np.linalg.norm(change_position)*EN_SCALE
        self.time_spent += time_elapsed/1000 #(convert ms to s)
        # print(self.position, self.trajectory[-1])
        # energy reduce as a function of distance moved
    
    def get_angle_diff(self):
        # helper function to get the angle between current orientation and the
        # straightline that connect platform center and self position
        a = np.array(self.platform) - np.array(self.position)
        radian = math.radians(self.orientation)
        # the coordinate have inverse y-axis, so its -cos and +sin
        b = np.array([-math.cos(radian), math.sin(radian)])
        cos_angle = np.inner(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
        # then I should calculate the angle between the two vector
        angle = np.rad2deg(math.acos(cos_angle)) 
        # I don't understand but its 180 rather than 0 when the orientation is 
        # inline with the vector between platform and self. But lets just accept this.
        angle = 180 - angle
        #angle = self.orientation -
        # angle = math.atan2((y2-y1),(x2-x1))*(180/math.pi)
        return angle
    
    def get_game_states(self):
        state = []
        # try normalizing them to the range between 0-1 by dividing off the max value
        state.append(self.orientation/360) # normalized angle
        # state.append(self.position[0]/WIDTH)
        # state.append(self.position[1]/WIDTH) # position
        # state.append(self.speed/MAX_SPEED)
        # state.append(self.time_spent/MAX_EPISODE_TIME) # time spent on task
        # state.append(self.energy/100) # energy spent on task
        # state.append(self.platform[0]/WIDTH)
        # state.append(self.platform[1]/WIDTH)
        # dist_to_target = np.linalg.norm(np.array(self.position)-np.array(self.platform))
        # state.append(dist_to_target)
        angle_to_target = self.get_angle_diff()
        state.append(angle_to_target)
        return state

        

        
# import time
# # pygame.init()
# maze = WaterMazeAI()
# # The human version while loop, allows key_board control.
# # game loop; it will stuck for some reason, figure out later.

# while True:   
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False
#         elif event.type == pygame.KEYDOWN:
#             if event.key == pygame.K_LEFT:
#                 action = [1,0,0,0,0]
#             elif event.key == pygame.K_RIGHT:
#                 action = [0,1,0,0,0]
#             elif event.key == pygame.K_UP:
#                 action = [0,0,1,0,0]
#             elif event.key == pygame.K_DOWN:
#                 action = [0,0,0,1,0]
#         else:
#             action = [0,0,0,0,1]
#     reward, game_over = maze.play_step(action)
#     print(maze.get_angle_diff())
#     #print(maze.orientation)
#     # print(maze.get_angle_diff())
#     if game_over:
#         break
# time.sleep(2)
# pygame.quit()
    