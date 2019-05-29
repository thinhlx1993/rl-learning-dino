# INITIALIZATION: libraries, parameters, network...
import random  # For sampling batches from the observations
import numpy as np

from keras.models import Sequential  # One layer after the other
from keras.layers import Dense, Flatten, Input, Dropout, BatchNormalization
from collections import deque  # For storing moves
from keras.applications import MobileNetV2, NASNetMobile, NASNetLarge
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from envs import Controller


# Parameter
model_path = 'saved_model.h5'
observetime = 400  # Number of timesteps we will be acting on the game and observing results
epsilon = 1  # Probability of doing a random move
epsilon_decay = 0.995
epsilon_min = 0.01
gamma = 0.9  # Discounted future reward. How much we care about steps further in time
mb_size = 384  # Learning minibatch size
num_episode = 10000
action_space = 3  # 0 is go straight, 1 is turn left, 2 is turn right
state_size = ()
# Create network. Input is two consecutive game states, output is Q-values of the possible moves.
model = Sequential()
# input_tensor = Input(shape=(285, 110, 3))
# inception_model = NASNetMobile(include_top=False, weights='imagenet', input_tensor=input_tensor)
# model.add(inception_model)
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.25))
model.add(Dense(24, activation='relu', input_dim=4))
model.add(Dense(24, activation='relu'))
model.add(Dense(3, activation='linear'))

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
# for layer in inception_model.layers:
#     layer.trainable = False

model.compile(loss='mse', optimizer='adam')
# model.load_weights(model_path)
print(model.summary())
#
# FIRST STEP: Knowing what each action does (Observing)
env = Controller()  # Choose game (any in the gym should work)

for episode in range(num_episode):
    D = deque()  # Register where the actions will be stored
    # env.reset()  # Game begins
    observation, reward, done, _ = env.step(1)
    state = np.expand_dims(observation, axis=0)
    for t in range(observetime):
        if np.random.rand() <= epsilon:
            action = np.random.randint(0, action_space, size=1)[0]  # jump or not
            # print('random action: {}'.format(action))
        else:
            Q = model.predict(state)  # Q-values predictions
            print('Q model: {}, predict action: {}'.format(Q[0], np.argmax(Q[0])))
            action = np.argmax(Q[0])  # Move with highest Q-value is the chosen one

        # See state of the game, reward... after performing the action
        observation_new, reward, done, action_step = env.step(action, ai_control=True)
        if reward:
            # Update the input with the new state of the game
            state_new = np.expand_dims(observation_new, axis=0)
            D.append((state, action_step, reward, state_new, done))  # 'Remember' action and consequence
            state = state_new  # Update state
            if done:
                env.reset()  # Restart game if it's finished
                state = np.expand_dims(observation, axis=0)

    if len(D) >= mb_size:
        minibatch = random.sample(D, mb_size)  # Sample some moves
        inputs_shape = (mb_size,) + state.shape[1:]
        inputs = np.zeros(inputs_shape)
        targets = np.zeros((mb_size, action_space))

        for i in range(0, mb_size):
            state = minibatch[i][0]
            action = minibatch[i][1]
            reward = minibatch[i][2]
            state_new = minibatch[i][3]
            done = minibatch[i][4]

            # Build Bellman equation for the Q function
            inputs[i:i + 1] = state
            target = reward
            if not done:
                target = (reward + gamma *
                          np.amax(model.predict(state_new)[0]))
            target_f = model.predict(state)
            target_f[0][action] = target
            targets[i] = target_f[0]

        # split to training and testing
        X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.25)

        # Train network to output the Q function
        model.fit(X_train, y_train, batch_size=10, epochs=1, verbose=1)
        print(model.evaluate(X_test, y_test))
        model.save_weights(model_path)
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
            print('epsilon: {}'.format(epsilon))

    #  Play game after learning
    env.playgame(model)
