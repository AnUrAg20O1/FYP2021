import pygame
from collections import namedtuple
from enum import Enum
import math

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

        #initialize plane direction as moving straight up
        self.direction = Direction.UP

        self.plane = Point(self.w/2, self.h-BLOCK_SIZE)
        #self.score = 0
        self.enemyPlane = Point(BLOCK_SIZE, self.h/2)
        #self.spawnEnemyPlane()
        self.env_length =  600
        self.score = 0

    def play(self):
        #get user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    self.direction = Direction.LEFT40
                elif event.key == pygame.K_s:
                    self.direction = Direction.LEFT20
                elif event.key == pygame.K_d:
                    self.direction = Direction.UP
                elif event.key == pygame.K_f:
                    self.direction = Direction.RIGHT20
                elif event.key == pygame.K_g:
                    self.direction = Direction.RIGHT40   
       
        #move plane and enemy plane
        self.movePlane(self.direction)
        self.moveEnemyPlane()
        self.env_length-=1

        #check if too near and calculate score
        d = self.closeness() #implement score
        if d <= 5*BLOCK_SIZE:
            self.score-=1
        if d <= BLOCK_SIZE:
            self.score-=3 

        #check if done
        if self.env_length<=0:
            done = True
        else:
            done = False

        #update UI and clock
        self.drawStuff(d)
        self.clock.tick(SPEED)

        return done, self.score
         
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

    def movePlane(self, direction):
        y = self.plane.y
        x = self.plane.x
        if y<40 or x<40 or x> 880:
            self.direction = Direction.UP

        if direction == Direction.LEFT40:
            x-=BLOCK_SIZE
        elif direction == Direction.LEFT20:
            x-=BLOCK_SIZE/2
        elif direction == Direction.RIGHT20:
            x+=BLOCK_SIZE/2
        elif direction == Direction.RIGHT40:
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


if __name__ == '__main__':
    game = Plane()
    
    # game loop
    while True:
        done, score = game.play()
        if done == True:
            break

    print("final score", score)  

    pygame.quit()





        
