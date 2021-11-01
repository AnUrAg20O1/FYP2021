import random
import gym
import numpy as np
#libraries to create a deep learning model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.optimizers import Adam
#libraries for building agent with keras RL (reinforcement learning)
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

env = gym.make('CartPole-v0')
states = env.observation_space.shape[0]
actions = env.action_space.n
#print(states)  the states are cart position, cart velocity, pole angle, pole angular velocity
#print(actions) the 2 actions are move left and move right... .

episodes = 10
for episode in range(0,episodes):
    state = env.reset()
    done = False
    score = 0 

    while not done:
        #env.render()
        action = random.choice([0,1])  #move left or right randomly
        n_state, reward, done, info = env.step(action)
        score+=reward

#    print("episode ",episode," score ",score) #check scores as a result of random movements

def build_model(states,actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,states)))
    model.add(Dense(24,activation='relu'))
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

#dqn.save_weights('dqn_weights.h5f', overwrite=True)
dqn.load_weights('dqn_weights.h5f')

scores = dqn.test(env,nb_episodes=10, visualize=True)
print(np.mean(scores.history['episode_reward']))
