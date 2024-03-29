"""
Double Q Learning
- Q learning is mode free bootstrapped off policy learning
- Double Q learning deals with maximization bias
    - Some state with many actions, with q(s, a) = 0; Q(s, a) is uncertain and has a positive bias
    - Max of true values is zero, but the max of Q(s,a) is positive, hence we have a positive bias
- Due to using same samples to determine the maximizing action and its .values()
- Solve this by using Q1, Q2 to determine max action and value, and then switching

Algorithm:
1. Initialize alpha and epsilon
2. Initialize Q1(s,a) and Q2(s,a)
3. Loop over episodes
    1. Initialize S
    2. For each episode
        1. Choose a from s using epsilon greedy in Q1 + Q2
        2. Take action a, observe r, s'
        3. With 0.5 probability
            1. Q1 -> Q1 + alpha * [r + gamma*Q2(s', argmaxQ1(s',a)) - Q1]
            2. Q2 -> Q2 + alpha * [r + gamma*Q1(s', argmaxQ2(s',a)) - Q2]
        4. s -> s'
"""

import gym
import matplotlib.pyplot as plt
import numpy as np


def maxAction(Q1, Q2, state):
    values = np.array([Q1[state, a] + Q2[state, a] for a in range(2)])
    action = np.argmax(values)
    return action


# discretize the spaces
poleThetaSpace = np.linspace(-0.20943951, 0.20943951, 10)
poleThetaVelSpace = np.linspace(-4, 4, 10)
cartPosSpace = np.linspace(-2.4, 2.4, 10)
cartVelSpace = np.linspace(-4, 4, 10)


def getState(observation):
    cartX, cartXdot, cartTheta, cartThetadot = observation
    cartX = int(np.digitize(cartX, cartPosSpace))
    cartXdot = int(np.digitize(cartXdot, cartVelSpace))
    cartTheta = int(np.digitize(cartTheta, poleThetaSpace))
    cartThetadot = int(np.digitize(cartThetadot, poleThetaVelSpace))

    return (cartX, cartXdot, cartTheta, cartThetadot)


def plotRunningAverage(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(totalrewards[max(0, t-100):(t+1)])
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()


if __name__ == '__main__':

    env = gym.make('CartPole-v0')
    # model hyperparameters
    ALPHA = 0.1
    GAMMA = 1.0
    EPS = 1.0

    # construct state space
    states = []
    for i in range(len(cartPosSpace)+1):
        for j in range(len(cartVelSpace)+1):
            for k in range(len(poleThetaSpace)+1):
                for l in range(len(poleThetaVelSpace)+1):
                    states.append((i, j, k, l))

    Q1, Q2 = {}, {}
    for s in states:
        for a in range(2):
            Q1[s, a] = 0
            Q2[s, a] = 0

    numGames = 100000
    totalRewards = np.zeros(numGames)
    for i in range(numGames):
        if i % 5000 == 0:
            print('starting game ', i)
        done = False
        epRewards = 0
        observation = env.reset()
        while not done:
            s = getState(observation)
            rand = np.random.random()
            a = maxAction(Q1, Q2, s) if rand < (
                1-EPS) else env.action_space.sample()
            observation_, reward, done, info = env.step(a)
            epRewards += reward
            s_ = getState(observation_)
            rand = np.random.random()

            if rand <= 0.5:
                a_ = maxAction(Q1, Q1, s_)
                Q1[s, a] = Q1[s, a] + ALPHA * \
                    (reward + GAMMA*Q2[s_, a_] - Q1[s, a])
            elif rand > 0.5:
                a_ = maxAction(Q2, Q2, s_)
                Q2[s, a] = Q2[s, a] + ALPHA * \
                    (reward + GAMMA * Q1[s_, a_] - Q2[s, a])

            observation = observation_
        EPS -= 2/(numGames) if EPS > 0 else 0
        totalRewards[i] = epRewards

    #plt.plot(totalRewards, 'b--')
    # plt.show()
    plotRunningAverage(totalRewards)
