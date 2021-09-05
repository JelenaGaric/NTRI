import gym
from DeepQLearning.dqn_implementation import Agent
from utils import plot_learning_curve, plot_learning_epsilon
import numpy as np

if __name__ == '__main__':
    environment = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, actions_num=4,
                  eps_end=0.01, input_dims=[8], learning_rate=0.001)
    scores, eps_history = [], []

    filename = 'dqn_lunar_lander.png'
    figure_file = '../plots/' + filename

    n_games = 500

    for i in range(n_games):
        score = 0
        done = False
        observation = environment.reset()
        while not done:
            action = agent.choose_action(observation)
            new_observation, reward, done, info = environment.step(action)
            score += reward
            agent.store_transition(observation, action, reward, new_observation, done)
            agent.learn()
            observation = new_observation

        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print('episode ', i, 'score %.2f' % score,
              'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)

    x = [i+1 for i in range(n_games)]
    plot_learning_epsilon(x, scores, eps_history, filename)

