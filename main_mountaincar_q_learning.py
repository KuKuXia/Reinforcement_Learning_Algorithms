import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import wrappers
import pickle

pos_space = np.linspace(-1.2, 0.6, 12)
vel_space = np.linspace(-0.07, 0.07, 20)


def get_state(observation):
    pos, vel = observation
    pos_bin = int(np.digitize(pos, pos_space))
    vel_bin = int(np.digitize(vel, vel_space))

    return (pos_bin, vel_bin)


def maxAction(Q, state, actions=[0, 1, 2]):
    values = np.array([Q[state, a] for a in actions])
    action = np.argmax(values)

    return action


if __name__ == "__main__":

    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 1000
    n_games = 50000
    alpha = 0.1
    gamma = 0.99
    eps = 1
    load_Q = True

    action_space = [0, 1, 2]

    states = []
    for pos in range(13):
        for vel in range(21):
            states.append((pos, vel))

    if load_Q:
        pickle_in = open('./tmp/mountaincar/mountaincar_q_learning.pkl', 'rb')
        Q = pickle.load(pickle_in)
        print('Loaded the Q values from the file.')
    else:
        Q = {}
        for state in states:
            for action in action_space:
                Q[state, action] = 0

    # env = wrappers.Monitor(env, 'tmp/mountaincar', video_callable = lambda episode_id: True, force=True)

    score = 0
    total_rewards = np.zeros(n_games)
    for i in range(n_games):
        done = False
        obs = env.reset()
        state = get_state(obs)
        if i % 1000 == 0 and i > 0:
            print('episode ', i, 'score ', score, 'epsilon %.3f' % eps)
        score = 0
        while not done:
            action = np.random.choice([0, 1, 2]) if np.random.random() < eps \
                else maxAction(Q, state)
            obs_, reward, done, info = env.step(action)
            state_ = get_state(obs_)
            score += reward
            action_ = maxAction(Q, state_)
            Q[state, action] = Q[state, action] + \
                alpha*(reward + gamma*Q[state_, action_] - Q[state, action])
            state = state_
            env.render()
        total_rewards[i] = score

        eps = eps - 2/n_games if eps > 0.01 else 0.01

    mean_rewards = np.zeros(n_games)
    for t in range(n_games):
        mean_rewards[t] = np.mean(total_rewards[max(0, t - 50): (t + 1)])
    plt.plot(mean_rewards)
    plt.show()
    plt.savefig('./tmp/mountaincar/mountaincar_q_learning.png')

    f = open('./tmp/mountaincar/mountaincar_q_learning.pkl', 'wb')
    pickle.dump(Q, f)
    f.close()
