import gym
from Policy_Gradient_TF_FC import PolicyGradientAgent
from utils import plotLearning_no_epsilon
from gym import wrappers

if __name__ == "__main__":
    agent = PolicyGradientAgent(lr=0.0005, gamma=0.99)
    # pip install box2d-py
    env = gym.make('LunarLander-v2')
    score_history = []
    score = 0
    num_episodes = 2000

    env = wrappers.Monitor(env, 'tmp/lunar-lander',
                           video_callable=lambda episode_id: True, force=True)

    for i in range(num_episodes):
        print('episode:  ', i, 'score: ', score)
        done = False
        score = 0
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)

            agent.store_transition(observation, action, reward)
            observation = observation_
            score += reward

        score_history.append(score)
        agent.learn()
        # agent.save_checkpoint()
    filename = 'lunar-lander-alpha0005-64*64fc-newG.png'
    plotLearning_no_epsilon(
        score_history, filename=filename, window=25)
