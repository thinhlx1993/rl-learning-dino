# INITIALIZATION: libraries, parameters, network...
import time
import matplotlib.pyplot as plt
from keras.models import Sequential  # One layer after the other
from keras.layers import Dense, Flatten, \
    Conv2D  # Dense layers are fully connected layers, Flatten layers flatten out multidimensional inputs
from collections import deque  # For storing moves
from controler import Controller
import numpy as np
import gym  # To train our network

env = Controller()  # Choose game (any in the gym should work)

import random  # For sampling batches from the observations

# Create network. Input is two consecutive game states, output is Q-values of the possible moves.
model = Sequential()
model.add(Dense(1024, input_shape=(2, 9200), init='uniform', activation='relu'))
# model.add(Dense(20, input_shape=(2,) + env.observation_space.shape, init='uniform', activation='relu'))
model.add(Flatten())  # Flatten input so as to have no problems with processing
model.add(Dense(128, init='uniform', activation='relu'))
model.add(Dense(10, init='uniform', activation='relu'))
model.add(Dense(2, init='uniform', activation='linear'))  # Same number of outputs as possible actions

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
print(model.summary())
# FIRST STEP: Knowing what each action does (Observing)

# Parameters
D = deque()  # Register where the actions will be stored
observetime = 100  # Number of timesteps we will be acting on the game and observing results
epsilon = 0.7  # Probability of doing a random move
gamma = 0.9  # Discounted future reward. How much we care about steps further in time
mb_size = 100  # Learning minibatch size
num_episode = 10000

for episode in range(num_episode):
    env.reset()  # Game begins
    observation, reward, done, _ = env.step(1)
    # (Formatting issues) Making the observation the first element of a batch of inputs
    obs = np.expand_dims(observation, axis=0)
    # np.stack((np.expand_dims(observation.flatten(), axis=0), np.expand_dims(observation.flatten(), axis=0)),
    #          axis=0).shape
    state = np.stack((obs, obs), axis=1)

    for t in range(observetime):
        if np.random.rand() <= epsilon:
            action = np.random.randint(0, 2, size=1)[0]  # jump or not
        else:
            Q = model.predict(state)  # Q-values predictions
            action = np.argmax(Q)  # Move with highest Q-value is the chosen one

        # See state of the game, reward... after performing the action
        observation_new, reward, done, info = env.step(action)
        # print('action: {}, observation shape: {}, reward: {}, done: {}'.format(action, observation_new.shape, reward,
        #                                                                        done))
        obs_new = np.expand_dims(observation_new, axis=0)  # (Formatting issues)

        # Update the input with the new state of the game
        state_new = np.append(np.expand_dims(obs_new, axis=0), state[:, :1, :], axis=1)
        D.append((state, action, reward, state_new, done))  # 'Remember' action and consequence
        state = state_new  # Update state
        if done:
            env.reset()  # Restart game if it's finished

            # (Formatting issues) Making the observation the first element of a batch of inputs
            obs = np.expand_dims(observation, axis=0)
            state = np.stack((obs, obs), axis=1)

    minibatch = random.sample(D, mb_size)  # Sample some moves

    inputs_shape = (mb_size,) + state.shape[1:]
    inputs = np.zeros(inputs_shape)
    targets = np.zeros((mb_size, 2))

    for i in range(0, mb_size):
        state = minibatch[i][0]
        action = minibatch[i][1]
        reward = minibatch[i][2]
        state_new = minibatch[i][3]
        done = minibatch[i][4]

        # Build Bellman equation for the Q function
        inputs[i:i + 1] = np.expand_dims(state, axis=0)
        targets[i] = model.predict(state)
        Q_sa = model.predict(state_new)

        if done:
            targets[i, action] = reward
        else:
            targets[i, action] = reward + gamma * np.max(Q_sa)

    # Train network to output the Q function
    history = model.train_on_batch(inputs, targets)
    print('loss: {}, acc: {}'.format(history[0], history[1]))
    model.save('saved_model.h5')
