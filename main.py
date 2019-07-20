# INITIALIZATION: libraries, parameters, network...
import random  # For sampling batches from the observations
import numpy as np

from collections import deque  # For storing moves
from sklearn.model_selection import train_test_split

from envs import Controller


# Parameter
model_path = 'saved_model.h5'
observetime = 1000  # Number of timesteps we will be acting on the game and observing results
epsilon = 1  # Probability of doing a random move
epsilon_decay = 0.995
epsilon_min = 0.01
gamma = 0.9  # Discounted future reward. How much we care about steps further in time
mb_size = 900  # Learning minibatch size
num_episode = 10000
action_space = 3  # 0 is go straight, 1 is turn left, 2 is turn right
state_size = 4

# FIRST STEP: Knowing what each action does (Observing)
env = Controller()  # Choose game (any in the gym should work)

for episode in range(num_episode):
    D = deque()  # Register where the actions will be stored
    # env.reset()  # Game begins
    observation, reward, done, _ = env.step(1)
    state = observation
    for t in range(observetime):
        if np.random.rand() <= epsilon:
            action = np.random.randint(0, action_space, size=1)[0]  # jump or not
            # print('random action: {}'.format(action))
        else:
            Q = env.actor.predict(state)  # Q-values predictions
            print('Q model: {}, predict action: {}'.format(Q[0], np.argmax(Q[0])))
            action = np.argmax(Q[0])  # Move with highest Q-value is the chosen one

        # See state of the game, reward... after performing the action
        observation_new, reward, done, action_step = env.step(action, ai_control=True)
        if reward:
            state_new = observation_new
            # Update the input with the new state of the game
            D.append((state, action, reward, state_new, done))  # 'Remember' action and consequence
            state = state_new  # Update state
            if done:
                env.reset()  # Restart game if it's finished

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
                target = (reward + gamma * np.amax(env.actor.predict(state_new)[0]))
            target_f = env.actor.predict(state)
            target_f[0][action] = target
            targets[i] = target_f[0]

        # split to training and testing
        X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.25)

        # Train network to output the Q function
        env.actor.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1)
        print(env.actor.evaluate(X_test, y_test))
        env.actor.save_weights(model_path)
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
            print('epsilon: {}'.format(epsilon))

    #  Play game after learning
    env.playgame()
