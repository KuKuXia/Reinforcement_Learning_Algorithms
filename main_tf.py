import gym
import numpy as np
from gym import wrappers

from DQN_TF import Agent, DeepQNetwork
from utils import plotLearning


def preprocess(observation):
    return np.mean(observation[30:, :], axis=2).reshape(180, 160, 1)


def stack_frames(stacked_frames, frame, buffer_size):
    if stacked_frames is None:
        stacked_frames = np.zeros((buffer_size, *frame.shape))
        for idx, _ in enumerate(stacked_frames):
            stacked_frames[idx, :] = frame
    else:
        stacked_frames[0:buffer_size-1, :] = stacked_frames[1:, :]
        stacked_frames[buffer_size - 1, :] = frame

    stacked_frames = stacked_frames.reshape(1, *frame.shape[0:2], buffer_size)
    return stacked_frames


if __name__ == "__main__":
    env = gym.make('Breakout-v0')
    load_checkpoint = False
    agent = Agent(gamma=0.99, epsilon=0.99, alpha=0.00025, input_dims=(
        180, 160, 4), n_actions=3, mem_size=25000, batch_size=32)
    if load_checkpoint:
        agent.load_models()

    filename = 'breakout-alpha0p000025-gamma0p99-only-one-fc.png'
    scores = []
    numGames = 3000
    eps_history = []
    stack_size = 4
    score = 0

    # uncomment the line below to record every episode
    env = wrappers.Monitor(env, 'tmp/breakout-v0',
                           video_callable=lambda episode_id: True, force=True)

    print("Loading up the agent's memory with random game play.")
    while agent.mem_counter < 25000:
        done = False
        observation = env.reset()
        observation = preprocess(observation)
        stacked_frames = None
        observation = stack_frames(stacked_frames, observation, stack_size)
        while not done:
            action = np.random.choice(([0, 1, 2]))
            action += 1
            observation_, reward, done, info = env.step(action)
            observation_ = stack_frames(
                stacked_frames, preprocess(observation_), stack_size)

            action -= 1
            agent.store_transition(observation, action,
                                   reward, observation_, int(done))

            observation = observation_

    print('Done with random game play, game on.')
    for i in range(numGames):
        done = False
        if i % 10 == 0 and i > 0:
            avg_score = np.mean(scores[max(0, i - 10): (i + 1)])
            print('episode', i, 'score', score, 'average_score %.3f' % avg_score,
                  'epsilon %.3f' % agent.epsilon)
            agent.save_models()
        else:
            print('episode: ', i, 'score', score)

        observation = env.reset()
        observation = preprocess(observation)
        stacked_frames = None
        observation = stack_frames(stacked_frames, observation, stack_size)

        score = 0
        while not done:
            action = agent.choose_action(observation)
            action += 1
            observation_, reward, done, info = env.step(action)
            observation_ = stack_frames(
                stacked_frames, preprocess(observation_), stack_size)

            action -= 1
            agent.store_transition(observation, action,
                                   reward, observation_, int(done))

            observation = observation_
            score += reward
            agent.learn()
        eps_history.append(agent.epsilon)
        scores.append(score)
    x = [i + 1 for i in range(numGames)]
    plotLearning(x, scores, eps_history, filename)
