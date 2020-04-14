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
    config.env = UnityEnvironment(file_name='Reacher.app')
    config.brain_name = config.env.brain_names[0]
    config.state_size = config.env.brains[config.brain_name].vector_observation_space_size
    config.action_size = config.env.brains[config.brain_name].vector_action_space_size

    # actor = Actor(config.state_size, config.action_size, fc1_units=256, fc2_units=256)
    # actor.load_state_dict(torch.load('model_weights/actor.pth',map_location='cpu'))
    config.actor_fn =  lambda : Actor(config.state_size, config.action_size, fc1_units=256, fc2_units=256)
    config.actor_opt_fn = lambda params: optim.Adam(params, lr=3e-4)

    # critic=Critic(config.state_size, config.action_size, fc1_units=256, fc2_units=256)
    # critic.load_state_dict(torch.load('model_weights/critic.pth',map_location='cpu'))
    config.critic_fn =  lambda : Critic(config.state_size, config.action_size, fc1_units=256, fc2_units=256)
    config.critic_opt_fn = lambda params: optim.Adam(params, lr=3e-4)

    config.replay_fn = lambda: Replay(config.action_size, buffer_size=int(1e6), batch_size=128)
    config.noise_fn = lambda: OrnsteinUhlenbeck(config.action_size, mu=0., theta=0.15, sigma=0.05)
    

    config.actor_path = 'model_weights/actor.pth'
    config.critic_path = 'model_weights/critic.pth'

    config.mode = 'Test'
    agent = Agent(config)
    

    env=config.env


    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]


    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)
    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    while True:
        actions = agent.act(states) # select an action (for each agent)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))


if __name__ == '__main__':
    main()