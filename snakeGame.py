from typing import Collection
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init() 

#reset()
#reward()
#play and direction
#game iteration
#iscollision

class Direction(Enum):
    RIGHT=1
    LEFT=2
    UP=3
    DOWN=4

Point = namedtuple('Point','x, y')
blockSize = 20
speed = 50

#colours for the gamescreen
white = (255,255,255)
red = (200,0,0)
blue1 = (0,0,255)
blue2 = (0,100,255)
black = (0,0,0)

font = pygame.font.SysFont('arial', 25)


class SnakeGame():
    def __init__(self,width=600,height=400):
        self.width = width
        self.height = height

        self.display = pygame.display.set_mode((self.width,self.height)) #create game window
        pygame.display.set_caption("snake game")  #name of game window
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT #set initial direction
        self.head = Point(self.width/2 ,self.height/2)
        self.snake = [self.head, Point(self.head.x-blockSize, self.head.y), Point(self.head.x-2*blockSize, self.head.y)]
        self.score = 0
        self.food = None

        self.placefood()
        self.iteration = 0

    def placefood(self):
        x = random.randint(0,(self.width-blockSize)//blockSize)*blockSize
        y = random.randint(0,(self.height-blockSize)//blockSize)*blockSize
        self.food = Point(x,y)
        if self.food in self.snake:
            self.placefood()

    
    def play_step(self, action):
        self.iteration+=1
        #collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            

        #move snake

        self.move(action)  #update new head position
        self.snake.insert(0, self.head)

        #check if game over
        reward = 0
        game_over = False
        if self.collision() or self.iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
          

        #place new food or just move
        if self.head == self.food:
            self.score+=1
            reward = 10 
            self.placefood()
        else:
            self.snake.pop() 

        #update UI and clock
        self.updateUI()
        self.clock.tick(speed)

        #return game over and score
        
        return reward, game_over, self.score


    def collision(self, pt = None):
        if pt == None:
            pt =  self.head   

        #hit boundary
        if pt.x>self.width - blockSize or pt.x<0 or pt.y> self.height -blockSize or pt.y<0:
            return True

        #hit itself
        if pt in self.snake[1:]: #0 is head itself from 1 starts body
            return True

        return False


    def move(self, action):
        #[straight,right or left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        index = clock_wise.index(self.direction)

        if np.array_equal(action, [1,0,0]):
            new_direction = clock_wise[index]  #no change
        elif np.array_equal(action, [0,1,0]):
            new_index = (index + 1) % 4
            new_direction = clock_wise[new_index]
        else:           
            new_index = (index - 1) % 4
            new_direction = clock_wise[new_index]

        self.direction = new_direction
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x+=blockSize
        elif self.direction == Direction.LEFT:
            x-=blockSize
        elif self.direction == Direction.UP:
            y-=blockSize
        elif self.direction == Direction.DOWN:
            y+=blockSize

        self.head = Point(x,y) 

    
    def updateUI(self):
        self.display.fill(black)

        for pt in self.snake:
            pygame.draw.rect(self.display, blue1, pygame.Rect(pt.x, pt.y, blockSize, blockSize))
            pygame.draw.rect(self.display, blue2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, red, pygame.Rect(self.food.x, self.food.y, blockSize, blockSize))
        text = font.render("score: "+str(self.score), True, white)
        self.display.blit(text, [0,0])
        pygame.display.flip()
