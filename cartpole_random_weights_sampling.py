import gym
import numpy as np
from gym import wrappers


env = gym.make('CartPole-v0')

bestLength = 0
episode_lengths = []

best_weights = np.zeros(4)

for i in range(100):
    new_weights = np.random.uniform(-1.0, 1.0, 4)

    length = []

    for j in range(100):
        observation = env.reset()
        done = False
        cnt = 0

        while not done:
            # env.render()
            cnt += 1

            action = 1 if np.dot(observation, new_weights) > 0 else 0

            observation_, reward, done, _ = env.step(action)

            observation = observation_

            if done:
                break
        length.append(cnt)
    average_length = float(sum(length) / len(length))

    if average_length > bestLength:
        bestLength = average_length
        best_weights = new_weights
    episode_lengths.append(average_length)

    if i % 10 == 0:
        print('Best length is ', bestLength)

done = False
cnt = 0
env = wrappers.Monitor(env, 'tmp/CartPole-V0', force=True)
observation = env.reset()

while not done:
    cnt += 1

    action = 1 if np.dot(observation, best_weights) > 0 else 0

    observation_, reward, done, _ = env.step(action)

    observation = observation_
print('Game lasted ', cnt, ' moves')
print('The best weights are ', best_weights)
