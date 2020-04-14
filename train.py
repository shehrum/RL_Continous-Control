from unityagents import UnityEnvironment
import numpy as np

from collections import deque
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import copy
from ddpg_agent import Agent, Replay, OrnsteinUhlenbeck
from model import Actor, Critic


def run(agent):
    config = agent.config
    scores_deque = deque(maxlen=100)
    scores = []

    for episode in range(1, config.max_episodes + 1):
        agent.reset()
        score = 0

        env_info = config.env.reset(train_mode=True)[config.brain_name]
        state = env_info.vector_observations[0]

        for step in range(config.max_steps):
            action = agent.act(state)
            env_info = config.env.step(action)[config.brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            agent.step(state, action, reward, next_state, done)

            score += reward
            state = next_state

            if done:
                break

        scores.append(score)
        scores_deque.append(score)
        mean_score = np.mean(scores_deque)

        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(episode, mean_score, score))

        if mean_score >= config.goal_score:
            break

    torch.save(agent.online_actor.state_dict(), config.actor_path)
    torch.save(agent.online_critic.state_dict(), config.critic_path)

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(scores) + 1), scores)
    ax.set_ylabel('Score')
    ax.set_xlabel('Episode #')
    fig.savefig(config.scores_path)
    plt.show()


class Config:
    def __init__(self, seed):
        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)

        self.env = None
        self.brain_name = None
        self.state_size = None
        self.action_size = None
        self.actor_fn = None
        self.actor_opt_fn = None
        self.critic_fn = None
        self.critic_opt_fn = None
        self.replay_fn = None
        self.noise_fn = None
        self.discount = None
        self.target_mix = None

        self.max_episodes = None
        self.max_steps = None

        self.actor_path = None
        self.critic_path = None
        self.scores_path = None
        self.mode= None


def main():
    config = Config(seed=6)
    config.env = UnityEnvironment(file_name='environment/Reacher_Linux_NoVis/Reacher.x86_64')
    config.brain_name = config.env.brain_names[0]
    config.state_size = config.env.brains[config.brain_name].vector_observation_space_size
    config.action_size = config.env.brains[config.brain_name].vector_action_space_size

    config.actor_fn = lambda: Actor(config.state_size, config.action_size, fc1_units=256, fc2_units=256)
    config.actor_opt_fn = lambda params: optim.Adam(params, lr=3e-4)

    config.critic_fn = lambda: Critic(config.state_size, config.action_size, fc1_units=256, fc2_units=256)
    config.critic_opt_fn = lambda params: optim.Adam(params, lr=3e-4)

    config.replay_fn = lambda: Replay(config.action_size, buffer_size=int(1e6), batch_size=128)
    config.noise_fn = lambda: OrnsteinUhlenbeck(config.action_size, mu=0., theta=0.15, sigma=0.05)

    config.discount = 0.99
    config.target_mix = 1e-3

    config.max_episodes = int(1000)
    config.max_steps = int(1e6)
    config.goal_score = 30

    config.actor_path = 'model_weights/actor.pth'
    config.critic_path = 'model_weights/critic.pth'
    config.scores_path = 'results/scores.png'
    config.mode= 'Train'
    agent = Agent(config)
    run(agent)


if __name__ == '__main__':
    main()