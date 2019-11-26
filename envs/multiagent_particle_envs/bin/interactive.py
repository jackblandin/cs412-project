#!/usr/bin/env python
import time
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse

import numpy as np

from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple.py', help='Path of the scenario Python script.')
    args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = False)
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    # create interactive policies for each agent
    policies = [InteractivePolicy(env,i) for i in range(env.n)]
    # execution loop
    obs_n = env.reset()

    rewards = np.zeros(len(env.world.agents))

    print ('env.discrete_action_space:', env.discrete_action_space)

    while True:
        # query for action from each agent's policy
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        # render all agent views
        env.render()
        # display rewards

        new_rewards = np.zeros(len(env.world.agents))
        for i, agent in enumerate(env.world.agents):
            new_rewards[i] = env._get_reward(agent)
        if (np.abs(rewards - new_rewards) > .001).any():
            print(rewards - new_rewards)
            rewards = new_rewards
            for i, r in enumerate(rewards):
                print('agent {} reward: {:.3f}'.format(i, r))

