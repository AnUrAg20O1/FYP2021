import gym
import numpy as np

env = gym.make("MountainCar-v0")
#print(env.action_space.n)

# Q-Learning constants
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000

DISCRETE_OS_SIZE = [20, 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
#print(discrete_os_win_size)
#making the q-table

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(int))

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n])) #create random values between 0 and -2 and fill in an array of dimensions 20 by 20 by 3
#print(q_table.size)

env.reset()
discrete_state = get_discrete_state(env.reset())
# print(discrete_state)
# print(q_table[discrete_state])
done = False
while not done:
    action = np.argmax(q_table[discrete_state]) #grab the maximum q value for the starting (reset) state
    new_state, reward, done, _ = env.step(action) #runs until 200 steps, the limit of he environmrnt
    new_discrete_state = get_discrete_state(new_state) #new q values after taking action
    #print(reward, new_state)
    env.render()

    #new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

    # If simulation did not end yet after last step - update Q table
    if not done:

        # Maximum possible Q value in next step (for new state)
        max_future_q = np.max(q_table[new_discrete_state])

        # Current Q value (for current state and performed action)
        current_q = q_table[discrete_state + (action,)]

        # And here's our equation for a new Q value for current state and action
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        # Update Q table with new Q value
        q_table[discrete_state + (action,)] = new_q


    # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly
    elif new_state[0] >= env.goal_position: #at 0 we have the reward
        #q_table[discrete_state + (action,)] = reward
        q_table[discrete_state + (action,)] = 0

    discrete_state = new_discrete_state


env.close()