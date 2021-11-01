from pygame.sprite import collide_circle
import torch   
import random
import numpy as np
from collections import deque   #check later
from snakeGame import SnakeGame, Direction, Point
from snakeGameModel import Linear_QNet, Qtrainer 
from snakeGameHelper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  #control randomness exploration vs exploitation
        self.gamma = 0.9  #discount rate
        self.memory = deque(maxlen = MAX_MEMORY)  #popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = Qtrainer(self.model, lr=LR, gamma=self.gamma)

        #more TODO later

    def get_state(self, game):
        head = game.snake[0]
        point_left = Point(head.x -20, head.y)
        point_right = Point(head.x +20, head.y)
        point_up = Point(head.x, head.y-20)
        point_down = Point(head.x, head.y+20)

        dir_left = game.direction == Direction.LEFT
        dir_right = game.direction == Direction.RIGHT
        dir_up = game.direction == Direction.UP
        dir_down = game.direction == Direction.DOWN

        state = [\
            #danger straight
            (dir_right and game.collision(point_right)) or
            (dir_left and game.collision(point_left)) or
            (dir_up and game.collision(point_up)) or
            (dir_down and game.collision(point_down)),

            #danger right
            (dir_up and game.collision(point_right)) or
            (dir_down and game.collision(point_left)) or
            (dir_left and game.collision(point_up)) or
            (dir_right and game.collision(point_down)),

            #danger left
            (dir_down and game.collision(point_right)) or
            (dir_up and game.collision(point_left)) or
            (dir_right and game.collision(point_up)) or
            (dir_left and game.collision(point_down)),

            #move direction
            dir_right,
            dir_left,
            dir_down,
            dir_up,

            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y < game.head.y
        ]

        return np.array(state, dtype=int)
                


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  #pop left if max memory is reached
        

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  #list of tuples
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
         

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        #random moves exploration vs exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()
    while True:
        #get old state
        old_state = agent.get_state(game)

        #get move
        final_move = agent.get_action(old_state)

        #perform move and get new state
        reward, done, score = game.play_step(final_move)
        new_state = agent.get_state(game)

        #train short memory
        agent.train_short_memory(old_state, final_move, reward, new_state, done)

        #remember
        agent.remember(old_state, final_move, reward, new_state, done)

        if done:
            #train the long memory/ experience replay and plot result
            game.reset()
            agent.n_games +=1
            agent.train_long_memory

            if score>record:
                record=score
                agent.model.save()

            print("game", agent.n_games," score", score," record", record)

            plot_scores.append(score)
            total_score +=score
            mean_score = total_score/agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)




if __name__ == '__main__':
    train()

