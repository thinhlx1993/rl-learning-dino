# INITIALIZATION: libraries, parameters, network...
from keras.models import Sequential  # One layer after the other
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from collections import deque  # For storing moves
from envs import Controller
import numpy as np
from keras.applications import InceptionResNetV2
from keras.layers import Input
from sklearn.model_selection import train_test_split
import time
import random  # For sampling batches from the observations


model_path = 'saved_model.h5'

# Create network. Input is two consecutive game states, output is Q-values of the possible moves.
model = Sequential()
input_tensor = Input(shape=(160, 160, 3))
inception_model = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=input_tensor)
model.add(inception_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='linear'))

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in inception_model.layers:
    layer.trainable = False

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.load_weights(model_path)

# FIRST STEP: Knowing what each action does (Observing)

# Parameters
env = Controller()  # Choose game (any in the gym should work)
observetime = 500  # Number of timesteps we will be acting on the game and observing results
epsilon = 0.7  # Probability of doing a random move
gamma = 0.9  # Discounted future reward. How much we care about steps further in time
mb_size = 320  # Learning minibatch size
num_episode = 10000


for episode in range(num_episode):
    D = deque()  # Register where the actions will be stored
    # env.reset()  # Game begins
    observation, reward, done, _ = env.step(1)
    first_obs = np.expand_dims(observation, axis=0)
    state = first_obs
    for t in range(observetime):
        if np.random.rand() <= epsilon:
            action = np.random.randint(0, 3, size=1)[0]  # jump or not
        else:
            Q = model.predict(state)  # Q-values predictions
            print(Q, np.argmax(Q[0]))
            action = np.argmax(Q[0])  # Move with highest Q-value is the chosen one

        # See state of the game, reward... after performing the action
        observation_new, reward, done, _ = env.step(action)
        if reward:
            obs_new = np.expand_dims(observation_new, axis=0)  # (Formatting issues)

            # Update the input with the new state of the game
            state_new = obs_new
            D.append((state, action, reward, state_new, done))  # 'Remember' action and consequence
            state = state_new  # Update state
            if done:
                env.reset()  # Restart game if it's finished
                state = first_obs

    if len(D) > mb_size:
        minibatch = random.sample(D, mb_size)  # Sample some moves
        inputs_shape = (mb_size,) + state.shape[1:]
        inputs = np.zeros(inputs_shape)
        targets = np.zeros((mb_size, 3))

        for i in range(0, mb_size-1):
            state = minibatch[i][0]
            action = minibatch[i][1]
            reward = minibatch[i+1][2]
            state_new = minibatch[i][3]
            done = minibatch[i+1][4]

            Q_sa = model.predict(state)
            # Build Bellman equation for the Q function
            # input_data = np.expand_dims(state, axis=0)
            inputs[i: i+1] = state
            targets[i] = model.predict(state)

            if done:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + gamma * np.amax(Q_sa[0])

        X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.33)
        # Train network to output the Q function
        model.fit(X_train, y_train, epochs=10, verbose=1)
        print(model.evaluate(X_test, y_test))
        model.save_weights(model_path)

    #  Play game after learning
    env.reset()
    observation, reward, done, _ = env.step(1)
    observation = np.expand_dims(observation, axis=0)
    tot_reward = 0.0
    state = observation
    while not done:
        Q = model.predict(state)
        action = np.argmax(Q[0])
        print(Q, action)
        observation, reward, done, info = env.step(action)
        obs = np.expand_dims(observation, axis=0)
        state = obs
        tot_reward += reward
    print('Game ended! Total reward: {}'.format(tot_reward))
