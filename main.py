import pybullet
import pybullet_envs
import gym
from gym import wrappers
import numpy as np
from sac_implementation import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    # environment = gym.make('InvertedPendulumBulletEnv-v0') BipedalWalker-v3 LunarLander-v2
    environment = gym.make('BipedalWalker-v3')
    agent = Agent(input_dims=environment.observation_space.shape, env=environment,
                  actions_num=environment.action_space.shape[0])
    games_num = 300
    filename = 'inverted_pendulum.png'
    figure_file = 'plots/'+filename

    best_score = environment.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        environment.render(mode='human')
    env = wrappers.Monitor(environment, "tmp/lunar-lander-sac",
                             video_callable=lambda episode_id: True, force=True)
    for i in range(games_num):
        #environment.render()

        observation = environment.reset()
        done = False
        score = 0

        while not done:
            action = agent.choose_action(observation)
            new_observation, reward, done, info = environment.step(action)
            score += reward
            agent.store_transition_to_memory(observation, action, reward, new_observation, done)
            if not load_checkpoint:
                agent.learn()
            observation = new_observation
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, ' score %.1f' % score, ' avg_score %.1f', avg_score)

    environment.close()

    if not load_checkpoint:
        x = [i+1 for i in range(games_num)]
        plot_learning_curve(x, score_history, figure_file)



