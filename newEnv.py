import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple

import numpy as np
import random

#libraries to create a deep learning model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.optimizers import Adam
#libraries for building agent with keras RL (reinforcement learning)
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory



#gym spaces
# print(Discrete(3).sample()) #discrete values between 0 and 2
# print()
# print(Box(2,9,shape = (3,3)).sample()) #random floating point values between lower 2 and 9
# print()
# print(Dict({"height":Discrete(2),"speed":Box(0,10,shape = (2,))}).sample()) #combine multiple sapces

class NewEnv(Env):
    def __init__(self):
        self.action_space = Discrete(3)  #action space
        self.observation_space = Box(0, 100, shape=(1,)) #observation space
        self.state = 38+random.randint(-4,4)
        self.env_length = 60


    def step(self,action):
        self.state+=action-1 #take action
        self.env_length-=1 #reduce time remaining
        #calculating reward
        if self.state >=37 and self.state <=39:
            reward = 1
        else:
            reward = -1

        #check if done
        if self.env_length<=0:
            done = True
        else:
            done = False
        
        self.state+=random.randint(-2,2)
        info = {}

        return self.state, reward, done, info


             
    def render(self):
        pass
    def reset(self):
        self.state = 38+random.randint(-4,4)
        self.env_length =60
        return self.state


env = NewEnv()
#print(env.observation_space.sample()) #sampling the actions in the environment randomly 

episodes = 10
for episode in range(0,episodes):
    state = env.reset()
    done = False
    score = 0 

    while not done:
        #env.render()
        action = env.action_space.sample()  #move left or right randomly
        n_state, reward, done, info = env.step(action)
        score+=reward
    #print("episode ",episode," score ",score)


    actions = env.action_space.n
    states = env.observation_space.shape

def build_model(states,actions):
    model = Sequential()
    model.add(Dense(24,activation='relu', input_shape = states))
    model.add(Dense(24,activation='relu'))
    model.add(Dense(actions,activation='linear'))
    return model    

model = build_model(states,actions)
#model.summary()

def build_agent(model,actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)

    return dqn

dqn = build_agent(model,actions)      
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
#dqn.fit(env, nb_steps = 50000, visualize=False, verbose=1)
#dqn.save_weights('dqn_weights_newEnv.h5f', overwrite=True)

dqn.load_weights('dqn_weights_newEnv.h5f')

scores = dqn.test(env,nb_episodes=50, visualize=False)
print(np.mean(scores.history['episode_reward']))