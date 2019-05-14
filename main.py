# INITIALIZATION: libraries, parameters, network...
from keras.models import Sequential  # One layer after the other
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from collections import deque  # For storing moves
from controler import Controller
from keras.models import load_model
import numpy as np
import random  # For sampling batches from the observations

env = Controller()  # Choose game (any in the gym should work)
model_path = 'saved_model.h5'


# Create network. Input is two consecutive game states, output is Q-values of the possible moves.
model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(125, 460, 4)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy', 'mse'])
# model.load_weights(model_path)

# FIRST STEP: Knowing what each action does (Observing)

# Parameters
observetime = 200  # Number of timesteps we will be acting on the game and observing results
epsilon = 0.6  # Probability of doing a random move
gamma = 0.9  # Discounted future reward. How much we care about steps further in time
mb_size = 32  # Learning minibatch size
num_episode = 10000

for episode in range(num_episode):
    D = deque()  # Register where the actions will be stored
    env.reset()  # Game begins
    observation, reward, done, _ = env.step(1)
    # (Formatting issues) Making the observation the first element of a batch of inputs
    obs = np.expand_dims(observation, axis=0)
    # np.stack((np.expand_dims(observation.flatten(), axis=0), np.expand_dims(observation.flatten(), axis=0)),
    #          axis=0).shape
    # state = np.stack((obs, obs), axis=1)
    state = obs
    for t in range(observetime):
        if np.random.rand() <= epsilon:
            action = np.random.randint(0, 2, size=1)[0]  # jump or not
        else:
            Q = model.predict(state)  # Q-values predictions
            action = int(Q[0][0])  # Move with highest Q-value is the chosen one

        # See state of the game, reward... after performing the action
        observation_new, reward, done, info = env.step(action)
        if reward:
            obs_new = np.expand_dims(observation_new, axis=0)  # (Formatting issues)

            # Update the input with the new state of the game
            state_new = obs_new
            D.append((state, action, reward, state_new, done))  # 'Remember' action and consequence
            state = state_new  # Update state
            if done:
                env.reset()  # Restart game if it's finished

                # (Formatting issues) Making the observation the first element of a batch of inputs
                obs = np.expand_dims(observation, axis=0)
                state = obs
    if len(D) > mb_size:
        minibatch = random.sample(D, mb_size)  # Sample some moves
        inputs_shape = (mb_size,) + state.shape[1:]
        inputs = np.zeros(inputs_shape)
        targets = np.zeros((mb_size, 1))

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
                targets[i] = reward + gamma * float(Q_sa[0][0])
            else:
                targets[i] = reward

        # Train network to output the Q function
        history = model.train_on_batch(inputs, targets)
        print('loss: {}, acc: {}, mse: {}'.format(history[0], history[1], history[2]))
        model.save_weights(model_path)
