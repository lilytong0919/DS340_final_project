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

pygame.init()


# a Action class that stores all possible actions
class MOVES:
    FASTER = 0
    SLOWER = 1
    CLOCKWISE = 2
    COUNTER_CLOCKWISE = 3
    

# Set RGB Color needed
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0,0,0) # potentially plot a trajectory
GRAY = (100,100,100)
BLUE = (0,70,120)

# unchanging part of the game
FPS = 60
PLATFORM = [300,350]
PLATFORM_RADIUS = 50
WIDTH = 800
HEIGHT = 800
POOL_CENTER = [400,400]
POOL_RADIUS = 375
MAX_SPEED = 400
FONT = pygame.font.SysFont('arial', 20)

# limit game length
MAX_EPISODE_TIME = 30

# the game class
class WaterMazeAI:
    def __init__(self, w = WIDTH, h = HEIGHT):
        self.width = w
        self.height = h
        # set up the display window?
        self.display = pygame.display.set_mode((self.width,self.height))
        pygame.display.set_caption('Water Maze')
        # not sure what clock do, but lets set it first
        self.clock = pygame.time.Clock()
        # platform location is fixed so I will set it here.
        self.platform = [300,350] # the center point of plat form
        self.reset()
        
    def reset(self):
        # upon reset the game, we should place the 'mouse' at a random location
        # facing a random direction
        self.speed = 0 
        self.position = [0,0]
        while not self.is_in_circle(POOL_CENTER,POOL_RADIUS,self.position):
            self.position = [random.randint(0,WIDTH),
                             random.randint(0,HEIGHT)]
            
        self.orientation = random.randint(0,360) # orientation in degree?
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
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        # 2 move
        self.move(action)
        # 3 check if game is over
        if self.is_game_over():
            reward = -20 # a small punishment for failing the task
            if self.is_on_platform():
                reward = 100 + self.energy # Reward saving energy
            game_over = True            
            # else:
            #     reward = -20
        # lets just start simple with -1 reward every iteration where the
        # game is not ended (ultimately I want it to tie with distance moved 
        # with an extra pushniment of using fast speed because moving fast is 
        # supposed to be tired!)
        # 4 I dont need to do anything I am just copying the structure
        # 5 update ui and clock
        self.update_ui()
        # I know, it's wired but I updated clock elsewhere so DONT DO SHIT HERE!
        # 6 return values
        return reward, game_over
    
    def update_ui(self):
        
        # update the display?
        self.display.fill(WHITE)        
        # Platform should be invisible to the AI, but lets plot it for now
        pygame.draw.circle(self.display, BLUE, POOL_CENTER, POOL_RADIUS + 20)
        pygame.draw.circle(self.display, GRAY, PLATFORM, PLATFORM_RADIUS)
        
        # text information here:
        text_lines = [FONT.render("Energy: " + str(self.energy), True, BLACK),
                      FONT.render("Time_spent: " + str(self.time_spent), True, BLACK)]
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
        if action == MOVES.FASTER:
            self.speed += 10
        elif action == MOVES.SLOWER:
            self.speed -= 10
        elif action == MOVES.CLOCKWISE:
            self.orientation += 2
        elif action == MOVES.COUNTER_CLOCKWISE:
            self.orientation -= 2
        
        # its fine to not do this step, but I feel like its good to keep the 
        # orientation within the 0-360 range
        if self.orientation < 0 or self.orientation > 360:
            self.orientation %= 360
        # keep speed in control
        if self.speed > MAX_SPEED:
            self.speed = MAX_SPEED
        elif self.speed < 0:
            self.speed = 0
        
        # I will put position calculation here for now.
        radian = math.radians(self.orientation)
        direction = np.array([math.cos(radian), -math.sin(radian)])
        time_elapsed = self.clock.tick(FPS)
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
        self.energy -= np.linalg.norm(change_position)*0.005
        self.time_spent += time_elapsed/1000 #(convert ms to s)
        # energy reduce as a function of distance moved
        

        
# import time
# maze = WaterMazeAI()
# # The human version while loop, allows key_board control.
# # game loop; it will stuck for some reason, figure out later.
# while True:   
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False
#         elif event.type == pygame.KEYDOWN:
#             if event.key == pygame.K_LEFT:
#                 action = 2
#             elif event.key == pygame.K_RIGHT:
#                 action = 3
#             elif event.key == pygame.K_UP:
#                 action = 0
#             elif event.key == pygame.K_DOWN:
#                 action = 1
#         else:
#             action = 5
#     reward, game_over = maze.play_step(action)
#     if game_over:
#         break
# time.sleep(2)
# pygame.quit()
    