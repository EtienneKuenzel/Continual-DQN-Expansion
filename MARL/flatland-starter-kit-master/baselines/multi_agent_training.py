from collections import deque
from datetime import datetime
import os
import random
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from pprint import pprint
from flatland.utils.rendertools import RenderTool
import psutil
import csv
from matplotlib import pyplot as plt
import torch
import numpy as np


from flatland.envs.step_utils.states import TrainState
from flatland1.envs.rail_env import RailEnv
from flatland1.envs.rail_env_1 import RailEnv as RailEnv1
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.rail_generators import sparse_rail_generator, rail_from_grid_transition_map
from flatland1.utils.simple_rail import make_custom_rail,make_deadlock_training, make_pathfinding_track, make_simple_rail_with_alternatives, make_malfunction_training

from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from utils.timer import Timer
from utils.observation_utils import normalize_observation
from reinforcement_learning.dddqn_policy import DQNPolicy, Continual_DQN_Expansion
from reinforcement_learning.PPO import PPOPolicy, A2CPolicy,TD3Policy
from reinforcement_learning.ordered_policy import OrderedPolicy
#from reinforcement_learning.cc_transformer import CentralizedCriticModel, CcTransformer
import seaborn as sns
import pandas as pd
import time




"""
This file shows how to train multiple agents using areinforcement learning approach.
After training an agent, you can submit it straight away to the Flatland 3 challenge!

Agent documentation: https://flatland.aicrowd.com/tutorials/rl/multi-agent.html
Submission documentation: https://flatland.aicrowd.com/challenges/flatland3/first-submission.html
"""
##sparseline env with growing complexity
def create_rail_env_1(tree_observation):
    # Break agents from time to time
    malfunction_parameters = MalfunctionParameters(
        malfunction_rate=0,
        min_duration=20,
        max_duration=50
    )
    return RailEnv(
        width=18, height=16,
        rail_generator=sparse_rail_generator(
            max_num_cities=2,
            grid_mode=False,
            max_rails_between_cities=1,
            max_rail_pairs_in_city=0
        ),
        line_generator=sparse_line_generator(),
        number_of_agents=1,
        malfunction_generator=ParamMalfunctionGen(malfunction_parameters),
        obs_builder_object=tree_observation,
        random_seed=0
    )
def create_rail_env_2(tree_observation):
    # Break agents from time to time
    malfunction_parameters = MalfunctionParameters(
        malfunction_rate=1/1000,
        min_duration=20,
        max_duration=20
    )
    return RailEnv(
        width=28, height=28,
        rail_generator=sparse_rail_generator(
            max_num_cities=4,
            grid_mode=False,
            max_rails_between_cities=2,
            max_rail_pairs_in_city=3
        ),
        line_generator=sparse_line_generator(),
        number_of_agents=3,
        malfunction_generator=ParamMalfunctionGen(malfunction_parameters),
        obs_builder_object=tree_observation,
        random_seed=0
    )
def create_rail_env_3(tree_observation):
    # Break agents from time to time
    malfunction_parameters = MalfunctionParameters(
        malfunction_rate=1/1000,
        min_duration=20,
        max_duration=20
    )
    return RailEnv(
        width=40, height=40,
        rail_generator=sparse_rail_generator(
            max_num_cities=5,
            grid_mode=False,
            max_rails_between_cities=3,
            max_rail_pairs_in_city=4
        ),
        line_generator=sparse_line_generator(),
        number_of_agents=6,
        malfunction_generator=ParamMalfunctionGen(malfunction_parameters),
        obs_builder_object=tree_observation,
        random_seed=0
    )
def create_rail_env_4(tree_observation):
    # Break agents from time to time
    malfunction_parameters = MalfunctionParameters(
        malfunction_rate=1/1000,
        min_duration=30,
        max_duration=30
    )
    return RailEnv(
        width=52, height=52,
        rail_generator=sparse_rail_generator(
            max_num_cities=6,
            grid_mode=False,
            max_rails_between_cities=3,
            max_rail_pairs_in_city=4
        ),
        line_generator=sparse_line_generator(),
        number_of_agents=10,
        malfunction_generator=ParamMalfunctionGen(malfunction_parameters),
        obs_builder_object=tree_observation,
        random_seed=0
    )
def create_rail_env_5(tree_observation):
    # Break agents from time to time
    malfunction_parameters = MalfunctionParameters(
        malfunction_rate=1/1000,
        min_duration=10,
        max_duration=30
    )
    return RailEnv(
        width=64, height=64,
        rail_generator=sparse_rail_generator(
            max_num_cities=8,
            grid_mode=False,
            max_rails_between_cities=3,
            max_rail_pairs_in_city=4
        ),
        line_generator=sparse_line_generator(),
        number_of_agents=14,
        malfunction_generator=ParamMalfunctionGen(malfunction_parameters),
        obs_builder_object=tree_observation,
        random_seed=0
    )

def create_deadlock(tree_observation, height, length, prob, agents):
    rail, rail_map, optionals = make_deadlock_training(length, height, prob)
    malfunction_parameters = MalfunctionParameters(
        malfunction_rate=1/64,
        min_duration=1,
        max_duration=10
    )
    return RailEnv(width=rail_map.shape[1],
          height=rail_map.shape[0],
          rail_generator= rail_from_grid_transition_map(rail, optionals),
          line_generator=sparse_line_generator(),
          number_of_agents=agents,
          obs_builder_object=tree_observation,
            malfunction_generator=ParamMalfunctionGen(malfunction_parameters)
          )
def make_custom_training(tree_observation):
    rail, rail_map, optionals = make_custom_rail()
    malfunction_parameters = MalfunctionParameters(
        malfunction_rate=1/1000,
        min_duration=10,
        max_duration=30
    )
    return RailEnv(width=rail_map.shape[1],
          height=rail_map.shape[0],
          rail_generator= rail_from_grid_transition_map(rail, optionals),
          line_generator=sparse_line_generator(),
          number_of_agents=7,
          obs_builder_object=tree_observation,
            malfunction_generator=ParamMalfunctionGen(malfunction_parameters)
          )
def create_pathfinding(tree_observation, size):
    rail, rail_map, optionals = make_pathfinding_track(size)
    return RailEnv(width=rail_map.shape[1],
          height=rail_map.shape[0],
          rail_generator= rail_from_grid_transition_map(rail, optionals),
          line_generator=sparse_line_generator(),
          number_of_agents=1,
          obs_builder_object=tree_observation,
          )
def create_pathfinding_oldreward(tree_observation, size):
    rail, rail_map, optionals = make_pathfinding_track(size)
    return RailEnv1(width=rail_map.shape[1],
          height=rail_map.shape[0],
          rail_generator= rail_from_grid_transition_map(rail, optionals),
          line_generator=sparse_line_generator(),
          number_of_agents=1,
          obs_builder_object=tree_observation,
          )
def create_malfunction(tree_observation, agent):
    rail, rail_map, optionals = make_malfunction_training(agent, 20)
    malfunction_parameters = MalfunctionParameters(
        malfunction_rate=1/100,
        min_duration=100,
        max_duration=100
    )
    return RailEnv(width=rail_map.shape[1],
          height=rail_map.shape[0],
          rail_generator= rail_from_grid_transition_map(rail, optionals),
          line_generator=sparse_line_generator(),
          number_of_agents=agent,
          malfunction_generator=ParamMalfunctionGen(malfunction_parameters),
          obs_builder_object=tree_observation,
          )

def epsilon(x):
    if x == 0:
        return 1
    else:
        return 1/(1.000005**x)
def train_agent(train_params, policy, curriculum, render=False):
    reset = 0
    j = 0
    evaluation = False
    episodes = []
    # Observation parameters
    observation_tree_depth = 3
    observation_radius = 10
    observation_max_path_depth = 30

    # Set the seeds
    random.seed(0)

    # Observation builder
    tree_observation = TreeObsForRailEnv(max_depth=observation_tree_depth, predictor=ShortestPathPredictorForRailEnv(observation_max_path_depth))

    # Setup the environments
    train_env = create_rail_env_5(tree_observation)


    # Calculate the state size given the depth of the tree observation and the number of features
    n_features_per_node = train_env.obs_builder.observation_dim
    n_nodes = sum([np.power(4, i) for i in range(observation_tree_depth + 1)])
    state_size = n_features_per_node * n_nodes
    action_size = 5

    # policy
    policy = policy(state_size, action_size, train_params)
    n_episodes = train_params.n_episodes
    a = 0
    for episode_idx in range(n_episodes):
        if curriculum == "1":
            train_env = create_rail_env_5(tree_observation)
        if curriculum == "2":
            train_env = create_rail_env_1(tree_observation)
            if j > 500:
                if a == 0:
                    print("Expansion")
                    policy.expansion()
                    a =1
                train_env = create_rail_env_2(tree_observation)
            if j > 1000:
                if a == 1:
                    print("Expansion")
                    policy.expansion()
                    a =2
                train_env = create_rail_env_3(tree_observation)
            if j > 1500:
                policy.reset_scores()
                evaluation = True
                train_env = make_custom_training(tree_observation)
            if j > 2000:
                policy.set_evaluation_mode(True)
                break
            """if j > 1500:
                if a == 2:
                    print("Expansion")
                    policy.expansion()
                    a =3
                train_env = create_rail_env_3(tree_observation)
            if j > 2000:
                if a == 2:
                    print("Expansion")
                    policy.expansion()
                    a =3
            if j > 600000:
                train_env = create_rail_env_4(tree_observation)"""
            if j > 800000:
                train_env = create_rail_env_5(tree_observation)
            if j > 1000000:
                train_env = create_rail_env_5(tree_observation)
        if curriculum == "3":
            train_env = create_pathfinding(tree_observation, 4)
            if j > 80000:
                train_env = create_pathfinding(tree_observation, 8)
            if j > 160000:
                train_env = create_pathfinding(tree_observation, 16)
            if j > 240000:
                train_env = create_pathfinding(tree_observation, 32)
            if j > 320000:
                if a == 0:
                    print("Expansion")
                    policy.expansion()
                    a = 1
                train_env = create_malfunction(tree_observation, 5)
            if j > 400000:
                train_env = create_malfunction(tree_observation, 6)
            if j > 480000:
                train_env = create_malfunction(tree_observation, 7)
            if j > 560000:
                train_env = create_malfunction(tree_observation, 8)
            if j > 640000:
                if a == 1:
                    print("Expansion")
                    policy.expansion()
                    a = 2
                train_env = create_deadlock(tree_observation, 2, 32, 100, 2)
            if j > 720000:
                train_env = create_deadlock(tree_observation, 2, 32, 80, 4)
            if j > 800000:
                train_env = create_deadlock(tree_observation, 4, 32, 60, 8)
            if j > 880000:
                train_env = create_deadlock(tree_observation, 4, 32, 50, 16)
            if j > 1000000:
                policy.reset_scores()
                evaluation = True
                train_env = make_custom_training(tree_observation)
            if j > 1075000:
                policy.set_evaluation_mode(True)
        if curriculum == "4":
            train_env = create_pathfinding(tree_observation, 4)
            if j > 65000:
                train_env = create_pathfinding(tree_observation, 8)
            if j > 130000:
                train_env = create_pathfinding(tree_observation, 16)
            if j > 195000:
                train_env = create_pathfinding(tree_observation, 32)
            if j > 260000:
                train_env = create_malfunction(tree_observation, 5)
            if j > 325000:
                train_env = create_malfunction(tree_observation, 6)
            if j > 390000:
                train_env = create_malfunction(tree_observation, 7)
            if j > 455000:
                train_env = create_malfunction(tree_observation, 8)
            if j > 520000:
                train_env = create_pathfinding(tree_observation, 32)
            if j > 585000:
                train_env = create_deadlock(tree_observation, 2, 32, 100, 2)
            if j > 650000:
                train_env = create_deadlock(tree_observation, 2, 32, 80, 4)
            if j > 715000:
                train_env = create_deadlock(tree_observation, 4, 32, 60, 8)
            if j > 780000:
                train_env = create_deadlock(tree_observation, 4, 32, 50, 16)
            if j > 845000:
                train_env = create_pathfinding(tree_observation, 32)
            if j > 910000:
                train_env = create_malfunction(tree_observation, 8)
            if j > 975000:
                train_env = create_deadlock(tree_observation, 4, 32, 50, 16)
            if j > 1000000:
                train_env = create_rail_env_5(tree_observation)

        if render: env_renderer = RenderTool(train_env, gl="PGL")


        # Reset environment
        obs, info = train_env.reset(regenerate_rail=True, regenerate_schedule=True)# schedule = false for custom

        # Init these values after reset()

        max_steps = train_env._max_episode_steps*3
        action_count = [0] * action_size
        action_dict = dict()
        n_agents = train_env.get_num_agents()
        agent_obs = [None] * n_agents
        agent_prev_obs = [None] * n_agents
        agent_prev_action = [2] * n_agents
        update_values = [False] * n_agents

        if render:env_renderer.set_new_rail()

        score = 0
        actions_taken = []

        # Build initial agent-specific observations
        for agent in train_env.get_agent_handles():
            if obs[agent]:
                agent_obs[agent] = normalize_observation(obs[agent], observation_tree_depth, observation_radius=observation_radius)
                agent_prev_obs[agent] = agent_obs[agent].copy()

        eps_start = epsilon(j)
        # Run episode
        for step in range(max_steps):

            for agent in train_env.get_agent_handles():
                if info['action_required'][agent]:
                    update_values[agent] = True
                    action = policy.act(agent, agent_obs[agent], eps=eps_start)
                    action_count[action] += 1
                    actions_taken.append(action)
                else:
                    # An action is not required if the train hasn't joined the railway network,
                    # if it already reached its target, or if is currently malfunctioning.
                    update_values[agent] = False
                    action = 0
                action_dict.update({agent: action})

            # Render an episode at some interval
            if render:env_renderer.render_env(show=True,frames=False,show_observations=False,show_predictions=True)


            # Update replay buffer and train agent-------------------------------------------------------------
            next_obs, all_rewards, done, info = train_env.step(action_dict)
            for agent in train_env.get_agent_handles():
                if update_values[agent] or done['__all__']:
                    # Only learn from timesteps where something happened
                    if not evaluation:
                        policy.step(agent, agent_prev_obs[agent], agent_prev_action[agent], all_rewards[agent] , agent_obs[agent], done[agent])
                    agent_prev_obs[agent] = agent_obs[agent].copy()
                    agent_prev_action[agent] = action_dict[agent]
                    j +=1
                # Preprocess the new observations
                if next_obs[agent]:
                    agent_obs[agent] = normalize_observation(next_obs[agent], observation_tree_depth, observation_radius=observation_radius)
                score += all_rewards[agent]
            if done['__all__']:
                break


        # Collect information about training
        tasks_finished = sum([agent.state == TrainState.DONE for agent in train_env.agents])
        completion = tasks_finished / max(1, train_env.get_num_agents())
        normalized_score = score / (max_steps * train_env.get_num_agents())
        episodes.append(j)

        d.get("networksteps").append(j)
        d.get("score").append(normalized_score)
        d.get("algo").append(policy.get_name())

        r.get("networksteps").append(j)
        r.get("completions").append(completion)
        r.get("algo").append(policy.get_name())
        policy.network_rotation(normalized_score)

        #a.append(train_env.return_agent_pos())
        if j >= 1150000:
            print("networksteps over 1  500 000")
            break
    a = 0
    for x in policy.get_expansion_code():
        a += 1
        for y in x:
            m.get("layer").append(a)
            m.get("type").append(y)




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-n", "--n_episodes", help="number of episodes to run", default= 100000000000, type=int)
    parser.add_argument("-t", "--training_env_config", help="training config id (eg 0 for Test_0)", default=0, type=int)
    parser.add_argument("-e", "--evaluation_env_config", help="evaluation config id (eg 0 for Test_0)", default=0, type=int)
    parser.add_argument("--n_evaluation_episodes", help="number of evaluation episodes", default=25, type=int)
    parser.add_argument("--checkpoint_interval", help="checkpoint interval", default=100, type=int)
    parser.add_argument("--eps_start", help="max exploration", default=1, type=float)
    parser.add_argument("--eps_end", help="min exploration", default=0.01, type=float)
    parser.add_argument("--eps_decay", help="exploration decay", default=0.9, type=float)
    parser.add_argument("--buffer_size", help="replay buffer size", default=int(1e6), type=int)
    parser.add_argument("--buffer_min_size", help="min buffer size to start training", default=0, type=int)
    parser.add_argument("--restore_replay_buffer", help="replay buffer to restore", default="", type=str)
    parser.add_argument("--save_replay_buffer", help="save replay buffer at each evaluation interval", default=False, type=bool)
    parser.add_argument("--batch_size", help="minibatch size", default=128, type=int)
    parser.add_argument("--gamma", help="discount factor", default=0.99, type=float)
    parser.add_argument("--tau", help="soft update of target parameters", default=1e-3, type=float)
    parser.add_argument("--learning_rate", help="learning rate", default=0.5e-4, type=float)
    parser.add_argument("--hidden_size", help="hidden size (2 fc layers)", default=512, type=int)
    parser.add_argument("--update_every", help="how often to update the network", default=8, type=int)
    parser.add_argument("--use_gpu", help="use GPU if available", default=True, type=bool)
    parser.add_argument("--num_threads", help="number of threads PyTorch can use", default=1, type=int)
    parser.add_argument("--render", help="render 1 episode in 100", default=False, type=bool)
    training_params = parser.parse_args()


    os.environ["OMP_NUM_THREADS"] = str(training_params.num_threads)
    policies = [Continual_DQN_Expansion]
    d = {'networksteps': [], 'algo': [], 'score': []}
    r = {'networksteps': [], 'algo': [], 'completions': []}
    m = {'layer': [], 'type': []}
    a=[]
    start_time = time.time()
    for x in policies:
        for y in ["3"]:
            for z in range(1):
                print("--------")
                print(x)
                print(y)
                print(z)
                train_agent(training_params, x, y)
                print("______________")

    with open("completions.csv", "w") as outfile1:
        writer = csv.writer(outfile1)
        writer.writerow(r.keys())
        writer.writerows(zip(*r.values()))

    with open("score.csv", "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(d.keys())
        writer.writerows(zip(*d.values()))
    with open("CDE.csv", "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(m.keys())
        writer.writerows(zip(*m.values()))
    print(time.time() - start_time)

    """with open("rail_usage.csv", 'w', newline='') as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(a)"""