import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import wrappers

from Policy_Gradient_Pytorch_fc import PolicyGradientAgent
from utils import plotLearning_no_epsilon

if __name__ == "__main__":
    agent = PolicyGradientAgent(ALPHA=0.001, input_dims=[
                                8], GAMMA=0.99, n_actions=4, layer1_size=128, layer2_size=128)
    # agent.load_checkpoint()
    env = gym.make('LunarLander-v2')
    score_history = []
    score = 0
    num_episode = 500

    env = wrappers.Monitor(env, 'tmp/lunar-lander',
                           video_callable=lambda episode_id: True, force=True)
    for i in range(num_episode):
        print('episode, ', i, ' score ', score)
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.store_rewards(reward)
            observation = observation_
            score += reward
        score_history.append(score)
        agent.learn()
        # agent.save_checkpoint()
    filename = './images/lunar-lander-alpha001-128*128fc-newG.png'
    plotLearning_no_epsilon(score_history, filename=filename, window=25)
