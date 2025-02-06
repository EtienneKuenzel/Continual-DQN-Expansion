
import os
import random
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from flatland.utils.rendertools import RenderTool

import csv
import numpy as np
import copy

from flatland.envs.step_utils.states import TrainState
from flatland1.envs.rail_env import RailEnv
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.rail_generators import sparse_rail_generator, rail_from_grid_transition_map
from flatland1.utils.simple_rail import make_custom_rail,make_deadlock_training, make_pathfinding_track, make_malfunction_training

from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from reinforcement_learning.dddqn_policy import Continual_DQN_Expansion, DQN_Policy, DQN_EWC_Policy, DQN_PAU_Policy, DQN_EWC_PAU_Policy, DQN_MAS_Policy,DQN_SI_Policy, DQN_CBP_Policy
from reinforcement_learning.ppo_a2c_policy import PPOPolicy, A2CPolicy
from observation_utils import normalize_observation

import time
import math
curriculum_steps = [(0, 'A'), (1, 'B'), (2, 'C')]

# Define the new positions
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
    speed_ratio_map = {
        0.8: 0.5,
        0.6: 0.00,
        0.4: 00.0,
        0.2: 0.5
    }
    return RailEnv(
        width=28, height=28,
        rail_generator=sparse_rail_generator(
            max_num_cities=4,
            grid_mode=False,
            max_rails_between_cities=2,
            max_rail_pairs_in_city=3
        ),
        line_generator=sparse_line_generator(speed_ratio_map),
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
    speed_ratio_map = {
        0.8: 0.50,
        0.6: 0.00,
        0.4: 0.25,
        0.2: 0.25
    }
    return RailEnv(
        width=40, height=40,
        rail_generator=sparse_rail_generator(
            max_num_cities=5,
            grid_mode=False,
            max_rails_between_cities=3,
            max_rail_pairs_in_city=4
        ),
        line_generator=sparse_line_generator(speed_ratio_map),
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
    speed_ratio_map = {
        0.8: 0.25,
        0.6: 0.25,
        0.4: 0.25,
        0.2: 0.25
    }
    return RailEnv(
        width=52, height=52,
        rail_generator=sparse_rail_generator(
            max_num_cities=6,
            grid_mode=False,
            max_rails_between_cities=3,
            max_rail_pairs_in_city=4
        ),
        line_generator=sparse_line_generator(speed_ratio_map),
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
    speed_ratio_map = {
        0.8: 0.25,
        0.6: 0.25,
        0.4: 0.25,
        0.2: 0.25
    }
    return RailEnv(
        width=64, height=64,
        rail_generator=sparse_rail_generator(
            max_num_cities=8,
            grid_mode=False,
            max_rails_between_cities=3,
            max_rail_pairs_in_city=4
        ),
        line_generator=sparse_line_generator(speed_ratio_map),
        number_of_agents=14,
        malfunction_generator=ParamMalfunctionGen(malfunction_parameters),
        obs_builder_object=tree_observation,
        random_seed=0
    )

def create_deadlock(tree_observation, height, length, prob, agents,x=1.0,y=0.0,c=0.0,v=0.0):
    rail, rail_map, optionals = make_deadlock_training(length, height, prob)
    malfunction_parameters = MalfunctionParameters(
        malfunction_rate=1/64,
        min_duration=1,
        max_duration=10
    )
    speed_ratio_map = {
        0.8: x,
        0.6: y,
        0.4: c,
        0.2: v
    }
    return RailEnv(width=rail_map.shape[1],
          height=rail_map.shape[0],
          rail_generator= rail_from_grid_transition_map(rail, optionals),
          line_generator=sparse_line_generator(speed_ratio_map),
          number_of_agents=agents,
          obs_builder_object=tree_observation,
          malfunction_generator=ParamMalfunctionGen(malfunction_parameters),
          name = "Deadlock" + str(agents)
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
            malfunction_generator=ParamMalfunctionGen(malfunction_parameters),
          name = "Evaluation"
          )
def create_pathfinding(tree_observation, size):
    rail, rail_map, optionals = make_pathfinding_track(size)
    return RailEnv(width=rail_map.shape[1],
          height=rail_map.shape[0],
          rail_generator= rail_from_grid_transition_map(rail, optionals),
          line_generator=sparse_line_generator(),
          number_of_agents=1,
          obs_builder_object=tree_observation,
          name="Pathfinding" + str(size)
          )
def create_malfunction(tree_observation, agent,x=1.0,y=0.0,c=0.0,v=0.0):
    rail, rail_map, optionals = make_malfunction_training(agent, 20)
    malfunction_parameters = MalfunctionParameters(
        malfunction_rate=1/100,
        min_duration=100,
        max_duration=100
    )
    speed_ratio_map = {
        0.8: x,
        0.6: y,
        0.4: c,
        0.2: v
    }
    return RailEnv(width=rail_map.shape[1],
          height=rail_map.shape[0],
          rail_generator= rail_from_grid_transition_map(rail, optionals),
          line_generator=sparse_line_generator(speed_ratio_map),
          number_of_agents=agent,
          malfunction_generator=ParamMalfunctionGen(malfunction_parameters),
          obs_builder_object=tree_observation,
          name="Malfunction" + str(agent)
          )
def eval_policy(tree_observation, policy, observation_tree_depth, observation_radius, networkstep):
    eval_env_list = [create_pathfinding(tree_observation, 32),create_malfunction(tree_observation, 8), create_deadlock(tree_observation, 4, 32, 50, 16),make_custom_training(tree_observation)]
    for env in eval_env_list:
        for x in range(20):
            obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)

            # Init these values after reset()
            max_steps = env._max_episode_steps * 3
            action_dict = dict()
            n_agents = env.get_num_agents()
            agent_obs = [None] * n_agents


            score = 0
            actions_taken = []

            # Build initial agent-specific observations
            for agent in env.get_agent_handles():
                if obs[agent]:
                    agent_obs[agent] = normalize_observation(obs[agent], observation_tree_depth,observation_radius=observation_radius)
            # Run episode
            for step in range(max_steps):
                for agent in env.get_agent_handles():
                    if info['action_required'][agent]:
                        action = policy.act(agent, agent_obs[agent], eps=0, eval=True)
                        actions_taken.append(action)
                    else:
                        action = 0
                    action_dict.update({agent: action})
                # Update Observation
                next_obs, all_rewards, done, info = env.step(action_dict)
                for agent in env.get_agent_handles():
                    # Preprocess the new observations
                    if next_obs[agent]:
                        agent_obs[agent] = normalize_observation(next_obs[agent], observation_tree_depth,observation_radius=observation_radius)
                    score += all_rewards[agent]
                if done['__all__']:
                    break
            tasks_finished = sum([agent.state == TrainState.DONE for agent in env.agents])
            completion = tasks_finished / max(1, env.get_num_agents())
            normalized_score = score / (max_steps * env.get_num_agents())

            q.get("networksteps").append(networkstep)
            q.get("score").append(normalized_score)
            q.get("completions").append(completion)
            q.get("algo").append(policy.get_name())
            q.get("env").append(env.name)
def epsilon(x):
    if x == 0:
        return 1
    else:
        return 1/(1.000005**x)
def write_to_csv(filename, data):
    with open(filename, "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data.keys())
        writer.writerows(zip(*data.values()))


def train_agent(train_params, policy):
    j = 0
    evaluation = False
    render = train_params.render
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


    # Calculate the state size given the deph tof the tree observation and the number of features
    state_size = train_env.obs_builder.observation_dim * sum([np.power(4, i) for i in range(observation_tree_depth + 1)])
    action_size = 5
    # policy
    policy = policy(state_size, action_size, train_params)
    expansion_done = set()
    while True:
        if training_params.curriculum == "no":
            train_env = create_rail_env_5(tree_observation)
            if j > 960000:
                train_env = make_custom_training(tree_observation)
                evaluation = True
        elif training_params.curriculum == "simple":
            train_env = create_rail_env_1(tree_observation)
            if j > 192000:
                train_env = create_rail_env_2(tree_observation)
            if j > 384000:
                train_env = create_rail_env_3(tree_observation)
            if j > 576000:
                train_env = create_rail_env_4(tree_observation)
            if j > 768000:
                train_env = create_rail_env_5(tree_observation)
            if j > 960000:
                train_env = make_custom_training(tree_observation)
                evaluation = True
        elif training_params.curriculum == "customPMD":
            expansion = [train_params.expansion * i for i in range(1, math.floor(train_params.envchange*12/train_params.expansion))]
            miniexpansion = [train_params.envchange * i for i in range(4, 12)]
            print(miniexpansion)
            for threshold in miniexpansion:
                if threshold in expansion:
                    if j >= threshold and threshold not in expansion_done:
                        policy.expansion()
                        expansion_done.add(threshold)
                elif j >= threshold and threshold not in expansion_done:
                    policy.miniexpansion()
                    expansion_done.add(threshold)
            curriculum_steps = [
                (train_params.envchange * 0, create_pathfinding(tree_observation, 4)),
                (train_params.envchange * 1, create_pathfinding(tree_observation, 8)),
                (train_params.envchange * 2, create_pathfinding(tree_observation, 16)),
                (train_params.envchange * 3, create_pathfinding(tree_observation, 32)),
                (train_params.envchange * 4, create_malfunction(tree_observation, 5,0.0,0.0,0.0,1.0)),
                (train_params.envchange * 5, create_malfunction(tree_observation, 6,0.5,0.0,0.0,0.5)),
                (train_params.envchange * 6, create_malfunction(tree_observation, 7,0.5,0.0,0.25,0.25)),
                (train_params.envchange * 7, create_malfunction(tree_observation, 8,0.25,0.25,0.25,0.25)),
                (train_params.envchange * 8, create_deadlock(tree_observation, 2, 32, 100, 2,0.,0.,0.,1.0)),
                (train_params.envchange * 9, create_deadlock(tree_observation, 2, 32, 80, 4,0.5,0.,0.,0.5)),
                (train_params.envchange * 10, create_deadlock(tree_observation, 4, 32, 60, 8,0.5,0.,0.25,0.25)),
                (train_params.envchange * 11, create_deadlock(tree_observation, 4, 32, 50, 16,0.25,0.25,0.25,0.25)),
                (train_params.envchange * 12, make_custom_training(tree_observation))]
            for threshold, env_func in curriculum_steps:
                if j >= threshold and threshold:
                    train_env = env_func
            if j !=0:
                if train_env.name != train_env_prev.name:
                    eval_policy(tree_observation, policy, observation_tree_depth, observation_radius, j)
            train_env_prev = train_env

            if j > train_params.envchange*12:
                evaluation = True
        elif training_params.curriculum == "customPDM":
            expansion = [train_params.expansion * i for i in range(1, math.floor(train_params.envchange*12/train_params.expansion))]
            for threshold in expansion:
                if j >= threshold and threshold not in expansion_done:
                    print(expansion)
                    policy.expansion()
                    expansion_done.add(threshold)
            curriculum_steps = [
                (train_params.envchange * 0, create_pathfinding(tree_observation, 4)),
                (train_params.envchange * 1, create_pathfinding(tree_observation, 8)),
                (train_params.envchange * 2, create_pathfinding(tree_observation, 16)),
                (train_params.envchange * 3, create_pathfinding(tree_observation, 32)),
                (train_params.envchange * 4, create_deadlock(tree_observation, 2, 32, 100, 2)),
                (train_params.envchange * 5, create_deadlock(tree_observation, 2, 32, 80, 4)),
                (train_params.envchange * 6, create_deadlock(tree_observation, 4, 32, 60, 8)),
                (train_params.envchange * 7, create_deadlock(tree_observation, 4, 32, 50, 16)),
                (train_params.envchange * 8, create_malfunction(tree_observation, 5)),
                (train_params.envchange * 9, create_malfunction(tree_observation, 6)),
                (train_params.envchange * 10, create_malfunction(tree_observation, 7)),
                (train_params.envchange * 11, create_malfunction(tree_observation, 8)),
                (train_params.envchange * 12, make_custom_training(tree_observation))]
            for threshold, env_func in curriculum_steps:
                if j >= threshold and threshold:
                    train_env = env_func
            if j !=0:
                if train_env.name != train_env_prev.name:
                    eval_policy(tree_observation, policy, observation_tree_depth, observation_radius, j)
            train_env_prev = train_env

            if j > train_params.envchange*12:
                evaluation = True
        elif training_params.curriculum == "customMDP":
            expansion = [train_params.expansion * i for i in range(1, math.floor(train_params.envchange*12/train_params.expansion))]
            for threshold in expansion:
                if j >= threshold and threshold not in expansion_done:
                    policy.expansion()
                    expansion_done.add(threshold)
            curriculum_steps = [
                (train_params.envchange * 0, create_malfunction(tree_observation, 5)),
                (train_params.envchange * 1, create_malfunction(tree_observation, 6)),
                (train_params.envchange * 2, create_malfunction(tree_observation, 7)),
                (train_params.envchange * 3, create_malfunction(tree_observation, 8)),
                (train_params.envchange * 4, create_deadlock(tree_observation, 2, 32, 100, 2)),
                (train_params.envchange * 5, create_deadlock(tree_observation, 2, 32, 80, 4)),
                (train_params.envchange * 6, create_deadlock(tree_observation, 4, 32, 60, 8)),
                (train_params.envchange * 7, create_deadlock(tree_observation, 4, 32, 50, 16)),
                (train_params.envchange * 8, create_pathfinding(tree_observation, 4)),
                (train_params.envchange * 9, create_pathfinding(tree_observation, 8)),
                (train_params.envchange * 10, create_pathfinding(tree_observation, 16)),
                (train_params.envchange * 11, create_pathfinding(tree_observation, 32)),
                (train_params.envchange * 12, make_custom_training(tree_observation))]
            for threshold, env_func in curriculum_steps:
                if j >= threshold and threshold:
                    train_env = env_func
            if j !=0:
                if train_env.name != train_env_prev.name:
                    eval_policy(tree_observation, policy, observation_tree_depth, observation_radius, j)
            train_env_prev = train_env

            if j > train_params.envchange*12:
                evaluation = True
        elif training_params.curriculum == "customMPD":
            expansion = [train_params.expansion * i for i in range(1, math.floor(train_params.envchange*12/train_params.expansion))]
            for threshold in expansion:
                if j >= threshold and threshold not in expansion_done:
                    policy.expansion()
                    expansion_done.add(threshold)
            curriculum_steps = [
                (train_params.envchange * 0, create_malfunction(tree_observation, 5)),
                (train_params.envchange * 1, create_malfunction(tree_observation, 6)),
                (train_params.envchange * 2, create_malfunction(tree_observation, 7)),
                (train_params.envchange * 3, create_malfunction(tree_observation, 8)),
                (train_params.envchange * 4, create_pathfinding(tree_observation, 4)),
                (train_params.envchange * 5, create_pathfinding(tree_observation, 8)),
                (train_params.envchange * 6, create_pathfinding(tree_observation, 16)),
                (train_params.envchange * 7, create_pathfinding(tree_observation, 32)),
                (train_params.envchange * 8, create_deadlock(tree_observation, 2, 32, 100, 2)),
                (train_params.envchange * 9, create_deadlock(tree_observation, 2, 32, 80, 4)),
                (train_params.envchange * 10, create_deadlock(tree_observation, 4, 32, 60, 8)),
                (train_params.envchange * 11, create_deadlock(tree_observation, 4, 32, 50, 16)),
                (train_params.envchange * 12, make_custom_training(tree_observation))]
            for threshold, env_func in curriculum_steps:
                if j >= threshold and threshold:
                    train_env = env_func
            if j !=0:
                if train_env.name != train_env_prev.name:
                    eval_policy(tree_observation, policy, observation_tree_depth, observation_radius, j)
            train_env_prev = train_env

            if j > train_params.envchange*12:
                evaluation = True
        elif training_params.curriculum == "customDPM":
            expansion = [train_params.expansion * i for i in range(1, math.floor(train_params.envchange*12/train_params.expansion))]
            for threshold in expansion:
                if j >= threshold and threshold not in expansion_done:
                    policy.expansion()
                    expansion_done.add(threshold)
            curriculum_steps = [
                (train_params.envchange * 0, create_deadlock(tree_observation, 2, 32, 100, 2)),
                (train_params.envchange * 1, create_deadlock(tree_observation, 2, 32, 80, 4)),
                (train_params.envchange * 2, create_deadlock(tree_observation, 4, 32, 60, 8)),
                (train_params.envchange * 3, create_deadlock(tree_observation, 4, 32, 50, 16)),
                (train_params.envchange * 4, create_pathfinding(tree_observation, 4)),
                (train_params.envchange * 5, create_pathfinding(tree_observation, 8)),
                (train_params.envchange * 6, create_pathfinding(tree_observation, 16)),
                (train_params.envchange * 7, create_pathfinding(tree_observation, 32)),
                (train_params.envchange * 8, create_malfunction(tree_observation, 5)),
                (train_params.envchange * 9, create_malfunction(tree_observation, 6)),
                (train_params.envchange * 10, create_malfunction(tree_observation, 7)),
                (train_params.envchange * 11, create_malfunction(tree_observation, 8)),
                (train_params.envchange * 12, make_custom_training(tree_observation))]
            for threshold, env_func in curriculum_steps:
                if j >= threshold and threshold:
                    train_env = env_func
            if j !=0:
                if train_env.name != train_env_prev.name:
                    eval_policy(tree_observation, policy, observation_tree_depth, observation_radius, j)
            train_env_prev = train_env

            if j > train_params.envchange*12:
                evaluation = True
        elif training_params.curriculum == "customDMP":
            expansion = [train_params.expansion * i for i in range(1, math.floor(train_params.envchange*12/train_params.expansion))]
            for threshold in expansion:
                if j >= threshold and threshold not in expansion_done:
                    policy.expansion()
                    expansion_done.add(threshold)
            curriculum_steps = [
                (train_params.envchange * 0, create_deadlock(tree_observation, 2, 32, 100, 2)),
                (train_params.envchange * 1, create_deadlock(tree_observation, 2, 32, 80, 4)),
                (train_params.envchange * 2, create_deadlock(tree_observation, 4, 32, 60, 8)),
                (train_params.envchange * 3, create_deadlock(tree_observation, 4, 32, 50, 16)),
                (train_params.envchange * 4, create_malfunction(tree_observation, 5)),
                (train_params.envchange * 5, create_malfunction(tree_observation, 6)),
                (train_params.envchange * 6, create_malfunction(tree_observation, 7)),
                (train_params.envchange * 7, create_malfunction(tree_observation, 8)),
                (train_params.envchange * 8, create_pathfinding(tree_observation, 4)),
                (train_params.envchange * 9, create_pathfinding(tree_observation, 8)),
                (train_params.envchange * 10, create_pathfinding(tree_observation, 16)),
                (train_params.envchange * 11, create_pathfinding(tree_observation, 32)),
                (train_params.envchange * 12, make_custom_training(tree_observation))]
            for threshold, env_func in curriculum_steps:
                if j >= threshold and threshold:
                    train_env = env_func
            if j !=0:
                if train_env.name != train_env_prev.name:
                    eval_policy(tree_observation, policy, observation_tree_depth, observation_radius, j)
            train_env_prev = train_env

            if j > train_params.envchange*12:
                evaluation = True
        elif training_params.curriculum == "customPMD+r":
            train_env = create_pathfinding(tree_observation, 4)
            if j > 65000:
                train_env = create_pathfinding(tree_observation, 8)
            if j > 130000:
                train_env = create_pathfinding(tree_observation, 16)
            if j > 195000:
                train_env = create_pathfinding(tree_observation, 32)
            if j > 260000:
                train_env = create_malfunction(tree_observation, 5, 0.,0.,0.,1.)
            if j > 325000:
                train_env = create_malfunction(tree_observation, 6, 0.5,0.,0.,0.5 )
            if j > 390000:
                train_env = create_malfunction(tree_observation, 7, 0.5,0.,0.25,0.25)
            if j > 455000:
                train_env = create_malfunction(tree_observation, 8, 0.25,0.25,0.25,0.25)
            if j > 520000:
                train_env = create_pathfinding(tree_observation, 32)
            if j > 585000:
                train_env = create_deadlock(tree_observation, 2, 32, 100, 2, 0.,0.,0.,1.0)
            if j > 650000:
                train_env = create_deadlock(tree_observation, 2, 32, 80, 4, 0.5,0.,0.,0.5)
            if j > 715000:
                train_env = create_deadlock(tree_observation, 4, 32, 60, 8, 0.5,0.,0.25,0.25)
            if j > 780000:
                train_env = create_deadlock(tree_observation, 4, 32, 50, 16, 0.25,0.25,0.25,0.25)
            if j > 845000:
                train_env = create_pathfinding(tree_observation, 32)
            if j > 910000:
                train_env = create_malfunction(tree_observation, 8, 0.25,0.25,0.25,0.25)
            if j > 960000:
                train_env = make_custom_training(tree_observation)
                evaluation = True
        else:
            print("Error: Non-Existent Curriculum")
            break

        if render: env_renderer = RenderTool(train_env, gl="PGL")
        # Reset environment
        obs, info = train_env.reset(regenerate_rail=True, regenerate_schedule=True)

        # Init these values after reset()
        max_steps = train_env._max_episode_steps*3
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
                    action = policy.act(agent, agent_obs[agent], eps=eps_start, eval=train_env.name=="Evaluation")#EDit
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
        d.get("env").append(train_env.name)

        r.get("networksteps").append(j)
        r.get("completions").append(completion)
        r.get("algo").append(policy.get_name())
        r.get("env").append(train_env.name)
        policy.network_rotation(normalized_score, completion)

        t.get("networksteps").append(j)
        t.get("function").append(policy.get_activation())
        t.get("type").append(policy.get_net())

        m['type'] = []
        m['score'] = []
        m['completions'] = []
        m.get('type').append(policy.networkEP)
        m.get('score').append(policy.networkEP_scores)
        m.get('completions').append(policy.networkEP_completions)
        for x in range(100):
            print("-----------------------")
        print(j)
        if j >= (train_params.envchange*12) + 100000:
            break




if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--render", help="render the environment", default=False, type=bool)

    parser.add_argument("--buffer_size", help="replay buffer size", default=int(1e6), type=int)
    parser.add_argument("--buffer_min_size", help="min buffer size to start training", default=0, type=int)
    parser.add_argument("--batch_size", help="minibatch size", default=128, type=int)
    parser.add_argument("--gamma", help="discount factor", default=0.99, type=float)
    parser.add_argument("--tau", help="soft update of target parameters", default=1e-3, type=float)
    parser.add_argument("--learning_rate", help="learning rate", default=0.5e-4, type=float)

    parser.add_argument("--anchors", help="CDE anchors", default=2, type=int)

    parser.add_argument("--ewc_lambda", help="impact of the weight locking", default=0.5, type=float)
    parser.add_argument("--hidden_size", help="neurons per layer", default=1024, type=int)
    parser.add_argument("--layer_count", help="count of layers", default=2, type=int)
    parser.add_argument("--update_every", help="how often to update the network", default=8, type=int)
    parser.add_argument("--use_gpu", help="use GPU if available", default=True, type=bool)
    parser.add_argument("--num_threads", help="number of threads PyTorch can use", default=1, type=int)

    parser.add_argument("--curriculum", help="choose a curriculum(replace ___ with PMD in any sequence)P=Pathfinding, M=Malfunction, D=Deadlock: custom___", default="customPMD", type=str)
    parser.add_argument("--envchange", help="time after environment change", default=80000, type=int)
    parser.add_argument("--expansion", help="time after expansion", default=320000, type=int)
    parser.add_argument("--policy", help="choose policy: CDE,DQN, EWC, PAU, PPO, A2C", default="CDE", type=str)
    parser.add_argument("--runs", help="repetitions of the training loop", default=1, type=int)
    training_params = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(training_params.num_threads)

    d = {'networksteps': [], 'algo': [], 'score': [], 'env': []}
    r = {'networksteps': [], 'algo': [], 'completions': [], 'env': []}
    t = {'networksteps': [],'function': [], 'type': []}
    q = {'networksteps': [], 'algo': [], 'score': [], 'completions':[], 'env': []}

    m = {'type' : [],'score' : [],'completions' : []}
    p = {'type': [], 'score': [], 'completions': []}
    a=[]
    policy_mapping = {
        "DQN": DQN_Policy,
        "CDE": Continual_DQN_Expansion,
        "EWC": DQN_EWC_Policy,
        "MAS": DQN_MAS_Policy,
        "SI": DQN_SI_Policy,
        "PAU": DQN_PAU_Policy,
        "CBP": DQN_CBP_Policy,
        "EWC+PAU": DQN_EWC_PAU_Policy,
        "PPO": PPOPolicy,
        "A2C": A2CPolicy
    }

    if policy_mapping.get(training_params.policy) is None:
        print("Error: Non-Existent Policy")
        sys.exit()
    print(policy_mapping.get(training_params.policy))

    start_time = time.time()
    for z in range(training_params.runs):
        print("Training Run: " + str(z))
        train_agent(training_params,policy_mapping.get(training_params.policy))
        for key in m:
            p[key].extend(m[key])

    print(time.time() - start_time)
    # safe data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_base_dir = os.path.join(script_dir, '..', '..', '..', 'Evaluation', 'neweval')
    os.makedirs(relative_base_dir, exist_ok=True)
    base_filename = f"{training_params.policy}-{training_params.layer_count}x{training_params.hidden_size}_{training_params.curriculum}"
    write_to_csv(os.path.join(relative_base_dir, f"completions_{base_filename}.csv"), r)
    write_to_csv(os.path.join(relative_base_dir, f"score_{base_filename}.csv"), d)
    write_to_csv(os.path.join(relative_base_dir, f"weights_{base_filename}.csv"), t)
    write_to_csv(os.path.join(relative_base_dir, f"expansion_{base_filename}.csv"), p)
    write_to_csv(os.path.join(relative_base_dir, f"eval_{base_filename}.csv"), q)
