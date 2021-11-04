import torch   
import random
import numpy as np
from collections import deque   #check later
from planeEnv import Plane, Direction, Point
from planeModel import Linear_QNet, Qtrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001   #learning rate

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 #control randomness
        self.gamma = 0.9  #discount rate
        self.memory = deque(maxlen = MAX_MEMORY)
        self.model = Linear_QNet(9, 256, 5)
        self.trainer = Qtrainer(self.model, lr=LR, gamma=self.gamma)

    
    def get_state(self, env):
        dir_left40 = env.direction == Direction.LEFT40
        dir_left20 = env.direction == Direction.LEFT20
        dir_right20 = env.direction == Direction.RIGHT20
        dir_up = env.direction == Direction.UP
        dir_right40 = env.direction == Direction.RIGHT40

        state = [\
            #move direction
            dir_right40,
            dir_right20,
            dir_left40,
            dir_left20,
            dir_up,

            env.plane.x < env.enemyPlane.x,
            env.plane.x > env.enemyPlane.x,
            env.plane.y < env.enemyPlane.y,
            env.plane.y > env.enemyPlane.y,
            
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
        #random moves
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0,0,0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,4)
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
    record = -100000
    agent = Agent()
    env = Plane()
    while True:
        #get old state
        state_old = agent.get_state(env)

        #get move
        final_move = agent.get_action(state_old)

        #perform move
        reward, done, score = env.play(final_move)
        state_new = agent.get_state(env)

        #train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        #rememebr
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            #train long memory
            env.reset()
            agent.n_games +=1
            agent.train_long_memory

            if score>record:
                record = score
                agent.model.save()

            print("game", agent.n_games," score", score," record", record)

            # plot_score.append(score)
            # total_score +=score
            # mean_score = total_score/agent.n_games
            # plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()