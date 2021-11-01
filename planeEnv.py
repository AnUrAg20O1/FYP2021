import pygame
from collections import namedtuple
from enum import Enum
import math
import numpy as np

pygame.init()
font = pygame.font.SysFont('arial', 20)

class Direction(Enum):
    LEFT20 = 1
    LEFT40 = 2
    RIGHT20 = 3
    RIGHT40 = 4
    UP = 5
    
Point = namedtuple('Point', 'x, y')
BLOCK_SIZE = 20
SPEED = 10

WHITE = (255, 255, 255)
BLACK = (0,0,0)
RED = (200,0,0)
BLUE = (0, 100, 255)

class Plane():
    def __init__(self, w=900, h=750):
        self.h = h
        self.w = w
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("plane")
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        #initialize plane direction as moving straight up
        self.direction = Direction.UP

        self.plane = Point(self.w/2, self.h-BLOCK_SIZE)
        self.enemyPlane = Point(BLOCK_SIZE, self.h/2)
        self.score = 0
        self.iteration = 0
        self.env_length = 150


    def play(self, action):
        #get user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
  
       
        #move plane and enemy plane
        self.movePlane(action)
        self.moveEnemyPlane()
        self.env_length-=1
        reward = 0 #need to be more elaborate to make sure less manoeuvers

        #check if too near and calculate score and reward
        d = self.closeness()
        if d <= 5*BLOCK_SIZE:
            self.score-=1
            reward = -30
        if d <= BLOCK_SIZE:
            reward = -100
            done = True
            return reward, done, self.score 

        #check if done
        if self.env_length<=0:
            done = True
        else:
            done = False

        #update UI and clock
        self.drawStuff(d)
        self.clock.tick(SPEED)

        return reward, done, self.score
         
    def closeness(self):
        self.display.fill(BLACK)
        y1 = self.plane.y
        x1 = self.plane.x
        y2 = self.enemyPlane.y
        x2 = self.enemyPlane.x
        d = math.sqrt((x1-x2)**2 + (y1-y2)**2)
        return d
    
    def drawStuff(self,d):
        self.display.fill(BLACK)
        pygame.draw.rect(self.display, BLUE, pygame.Rect(self.plane.x, self.plane.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.enemyPlane.x, self.enemyPlane.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.circle(self.display, RED, (self.enemyPlane.x+10, self.enemyPlane.y+10), 5*BLOCK_SIZE, 2)
        if d < 5*BLOCK_SIZE:
            text = font.render("collision!!!  score=" + str(self.score), True, WHITE)
        else:
            text = font.render("no collision!!!  score=" + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def movePlane(self, action):
        #[left40, left 20, straight, right20, right40]
    
        if np.array_equal(action, [1,0,0,0,0]):
            new_direction = Direction.LEFT40
        elif np.array_equal(action, [0,1,0,0,0]):
            new_direction = Direction.LEFT20
        elif np.array_equal(action,[0,0,1,0,0]):
            new_direction = Direction.UP
        elif np.array_equal(action,[0,0,0,1,0]):
            new_direction = Direction.RIGHT20
        else:
            new_direction = Direction.RIGHT40
        
        self.direction = new_direction

        y = self.plane.y
        x = self.plane.x
        if y<40 or x<40 or x> 880:
            self.direction = Direction.UP

        if self.direction == Direction.LEFT40:
            x-=BLOCK_SIZE
        elif self.direction == Direction.LEFT20:
            x-=BLOCK_SIZE/2
        elif self.direction == Direction.RIGHT20:
            x+=BLOCK_SIZE/2
        elif self.direction == Direction.RIGHT40:
            x+=BLOCK_SIZE

        y-=BLOCK_SIZE
        if y <= 20 or x<30 or x> 900:
            y = self.h-BLOCK_SIZE
            x = self.w/2
        self.plane = Point(x,y)    

    def moveEnemyPlane(self):
        y = self.enemyPlane.y
        x = self.enemyPlane.x
        x+=BLOCK_SIZE
        if x > 900:
            x = BLOCK_SIZE
        self.enemyPlane = Point(x,y) 
