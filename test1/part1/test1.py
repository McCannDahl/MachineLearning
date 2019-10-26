import gym
import numpy as np

# Tweekable values
DISCRETE_OS_SIZE = [20,20]
Q_HIGH = 0
Q_LOW = -2
# if discount is ~1, then long term rewards. if low, then prioritize short term
# Q-Learning settings
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 500
# Exploration settings
epsilon = 1  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

env = gym.make("MountainCar-v0")
print(env.action_space.n)
print(env.reset())
print(env.observation_space.high)
print(env.observation_space.low)

discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
print(discrete_os_win_size)
q_table = np.random.uniform(low=Q_LOW, high=Q_HIGH, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))  # we use this tuple to look up the 3 Q values for the available actions in the q-table

for episode in range(EPISODES):
    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(q_table[discrete_state])
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)
        new_state, reward, done, extra_info = env.step(action)
        #print(reward, new_state)
        new_discrete_state = get_discrete_state(new_state)

        if episode % SHOW_EVERY == 0:
            env.render()

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
        elif new_state[0] >= env.goal_position:
            #q_table[discrete_state + (action,)] = reward
            q_table[discrete_state + (action,)] = 0
        discrete_state = new_discrete_state
        # Decaying is being done every episode if episode number is within decaying range
        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value

env.close()