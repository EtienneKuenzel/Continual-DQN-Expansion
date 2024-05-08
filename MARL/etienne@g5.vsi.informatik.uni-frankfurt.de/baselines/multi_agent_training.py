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

from matplotlib import pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
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
from reinforcement_learning.dddqn_policy import DDDQNPolicy, DoubleDeepQNetworkPolicy, DuelingDeepQNetworkPolicy, DeepQNetworkPolicy, DDDQN_UCBPolicy
from reinforcement_learning.PPO import PPOPolicy, A2CPolicy,TD3Policy
from reinforcement_learning.ordered_policy import OrderedPolicy
#from reinforcement_learning.cc_transformer import CentralizedCriticModel, CcTransformer
import seaborn as sns
import pandas as pd
sns.set_theme(style="darkgrid")


"""
This file shows how to train multiple agents using a reinforcement learning approach.
After training an agent, you can submit it straight away to the Flatland 3 challenge!

Agent documentation: https://flatland.aicrowd.com/tutorials/rl/multi-agent.html
Submission documentation: https://flatland.aicrowd.com/challenges/flatland3/first-submission.html
"""
##sparseline env with growing complexity
def create_rail_env_1(env_params, tree_observation):
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
def create_rail_env_2(env_params, tree_observation):
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
def create_rail_env_3(env_params, tree_observation):
    # Break agents from time to time
    malfunction_parameters = MalfunctionParameters(
        malfunction_rate=1/1000,
        min_duration=20,
        max_duration=20
    )
    return RailEnv(
        width=40, height=40,
        rail_generator=sparse_rail_generator(
            max_num_cities=8,
            grid_mode=False,
            max_rails_between_cities=3,
            max_rail_pairs_in_city=4
        ),
        line_generator=sparse_line_generator(),
        number_of_agents=8,
        malfunction_generator=ParamMalfunctionGen(malfunction_parameters),
        obs_builder_object=tree_observation,
        random_seed=0
    )
def create_rail_env_4(env_params, tree_observation):
    # Break agents from time to time
    malfunction_parameters = MalfunctionParameters(
        malfunction_rate=1/1000,
        min_duration=30,
        max_duration=30
    )
    return RailEnv(
        width=52, height=52,
        rail_generator=sparse_rail_generator(
            max_num_cities=20,
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
def create_rail_env_5(env_params, tree_observation):
    # Break agents from time to time
    malfunction_parameters = MalfunctionParameters(
        malfunction_rate=1/1000,
        min_duration=10,
        max_duration=30
    )
    return RailEnv(
        width=64, height=64,
        rail_generator=sparse_rail_generator(
            max_num_cities=12,
            grid_mode=False,
            max_rails_between_cities=2,
            max_rail_pairs_in_city=3
        ),
        line_generator=sparse_line_generator(),
        number_of_agents=20,
        malfunction_generator=ParamMalfunctionGen(malfunction_parameters),
        obs_builder_object=tree_observation,
        random_seed=0
    )



def create_deadlock(tree_observation, size):
    rail, rail_map, optionals = make_deadlock_training(32, 2, 100)
    return RailEnv(width=rail_map.shape[1],
          height=rail_map.shape[0],
          rail_generator= rail_from_grid_transition_map(rail, optionals),
          line_generator=sparse_line_generator(),
          number_of_agents=4,
          obs_builder_object=tree_observation,
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
def create_malfunction(tree_observation, size):
    rail, rail_map, optionals = make_malfunction_training()
    malfunction_parameters = MalfunctionParameters(
        malfunction_rate=3/100,
        min_duration=100,
        max_duration=100
    )
    return RailEnv(width=rail_map.shape[1],
          height=rail_map.shape[0],
          rail_generator= rail_from_grid_transition_map(rail, optionals),
          line_generator=sparse_line_generator(),
          number_of_agents=5,
          malfunction_generator=ParamMalfunctionGen(malfunction_parameters),
          obs_builder_object=tree_observation,
          )


def train_agent(train_params, train_env_params, eval_env_params, obs_params, policy, curriculum, runcounter):
    j = 0
    sparse_scores = []
    sparse_completions = []
    episodes = []
    # Environment parameters
    n_agents = train_env_params.n_agents
    x_dim = train_env_params.x_dim
    y_dim = train_env_params.y_dim
    n_cities = train_env_params.n_cities
    max_rails_between_cities = train_env_params.max_rails_between_cities
    max_rail_pairs_in_city = train_env_params.max_rail_pairs_in_city
    seed = train_env_params.seed

    # Unique ID for this training
    now = datetime.now()
    training_id = now.strftime('%y%m%d%H%M%S')

    # Observation parameters
    observation_tree_depth = obs_params.observation_tree_depth
    observation_radius = obs_params.observation_radius
    observation_max_path_depth = obs_params.observation_max_path_depth

    # Training parameters
    eps_start = train_params.eps_start
    eps_end = train_params.eps_end
    eps_decay = train_params.eps_decay
    n_episodes = train_params.n_episodes
    checkpoint_interval = train_params.checkpoint_interval
    n_eval_episodes = train_params.n_evaluation_episodes
    restore_replay_buffer = train_params.restore_replay_buffer
    save_replay_buffer = train_params.save_replay_buffer

    # Set the seeds
    random.seed(seed)

    # Observation builder
    predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
    tree_observation = TreeObsForRailEnv(max_depth=observation_tree_depth, predictor=predictor)

    # Setup the environments

    train_env1 = create_rail_env_1(train_env_params, tree_observation)
    train_env2 = create_rail_env_2(train_env_params, tree_observation)
    train_env3 = create_rail_env_3(train_env_params, tree_observation)
    train_env4 = create_rail_env_4(train_env_params, tree_observation)
    train_env5 = create_rail_env_5(train_env_params, tree_observation)

    pathfinding_env_1 = create_pathfinding(tree_observation, 4)
    pathfinding_env_2 = create_pathfinding(tree_observation, 8)
    pathfinding_env_3 = create_pathfinding(tree_observation, 16)
    pathfinding_env_4 = create_pathfinding(tree_observation, 32)#
    malfunction_env_1 = create_malfunction(tree_observation, 1)
    malfunction_env_2 = create_pathfinding(tree_observation, 4)
    malfunction_env_3 = create_pathfinding(tree_observation, 8)
    malfunction_env_4 = create_pathfinding(tree_observation, 16)
    deadlock_env1 = create_deadlock(tree_observation, 8)
    train_env = train_env2


    # Calculate the state size given the depth of the tree observation and the number of features
    n_features_per_node = train_env.obs_builder.observation_dim
    n_nodes = sum([np.power(4, i) for i in range(observation_tree_depth + 1)])
    state_size = n_features_per_node * n_nodes
    # The action space of flatland is 5 discrete actions
    action_size = 5

    # Smoothed values used as target for hyperparameter tuning
    smoothed_normalized_score = -1.0
    smoothed_eval_normalized_score = -1.0
    smoothed_completion = 0.0
    smoothed_eval_completion = 0.0

    # policy
    policy = policy(state_size, action_size, train_params)


    #print("\n💾 Replay buffer status: {}/{} experiences".format(len(policy.memory.memory), train_params.buffer_size))

    hdd = psutil.disk_usage('/')
    if save_replay_buffer and (hdd.free / (2 ** 30)) < 500.0:
        print("⚠️  Careful! Saving replay buffers will quickly consume a lot of disk space. You have {:.2f}gb left.".format(hdd.free / (2 ** 30)))

    # TensorBoard writer
    writer = SummaryWriter()
    writer.add_hparams(vars(train_params), {})
    writer.add_hparams(vars(train_env_params), {})
    writer.add_hparams(vars(obs_params), {})

    training_timer = Timer()
    training_timer.start()

    print("\n🚉 Training {} trains on {}x{} grid for {} episodes, evaluating on {} episodes every {} episodes. Training id '{}'.\n".format(
        n_agents,
        x_dim, y_dim,
        n_episodes,
        n_eval_episodes,
        checkpoint_interval,
        training_id
    ))
    b = 0
    a = 0
    for episode_idx in range(n_episodes):
        if curriculum == "1":
            train_env = train_env4
        if curriculum == "2":
            train_env = train_env1
            if j > 200000:
                train_env = train_env2
            if j > 400000:
                train_env = train_env3
            if j > 600000:
                train_env = train_env4
        if curriculum == "3":
            train_env = pathfinding_env_1
            if j > 50000:
                train_env = pathfinding_env_2
            if j > 100000:
                train_env = pathfinding_env_3
            if j > 150000:
                train_env = pathfinding_env_4
            if j > 200000:
                train_env = malfunction_env_1
            if j > 400000:
                train_env = deadlock_env1
            if j > 600000:
                train_env = train_env4
            # Setup renderer

        if True:
            env_renderer = RenderTool(train_env, gl="PGL")
        step_timer = Timer()
        reset_timer = Timer()
        learn_timer = Timer()
        preproc_timer = Timer()
        inference_timer = Timer()

        # Reset environment
        reset_timer.start()

        obs, info = train_env.reset(regenerate_rail=True, regenerate_schedule=True)# schedule = false for custom
        reset_timer.end()

        # Init these values after reset()

        max_steps = train_env._max_episode_steps*2
        action_count = [0] * action_size
        action_dict = dict()
        agent_obs = [None] * n_agents
        agent_prev_obs = [None] * n_agents
        agent_prev_action = [2] * n_agents
        update_values = [False] * n_agents

        if False:
            env_renderer.set_new_rail()

        score = 0
        nb_steps = 0
        actions_taken = []

        # Build initial agent-specific observations
        for agent in train_env.get_agent_handles():
            if obs[agent]:
                agent_obs[agent] = normalize_observation(obs[agent], observation_tree_depth, observation_radius=observation_radius)
                agent_prev_obs[agent] = agent_obs[agent].copy()

        # Run episode
        for step in range(max_steps):
            inference_timer.start()
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
            inference_timer.end()

            # Environment step
            step_timer.start()

            next_obs, all_rewards, done, info = train_env.step(action_dict)

            step_timer.end()
            # Render an episode at some interval
            if False:
                env_renderer.render_env(
                    show=True,
                    frames=False,
                    show_observations=False,
                    show_predictions=True
                )

           # Update replay buffer and train agent
            for agent in train_env.get_agent_handles():
                if update_values[agent] or done['__all__']:
                    # Only learn from timesteps where something happened
                    learn_timer.start()

                    policy.step(agent, agent_prev_obs[agent], agent_prev_action[agent], all_rewards[agent] , agent_obs[agent], done[agent])
                    learn_timer.end()
                    agent_prev_obs[agent] = agent_obs[agent].copy()
                    agent_prev_action[agent] = action_dict[agent]
                    j +=1
                # Preprocess the new observations
                if next_obs[agent]:
                    preproc_timer.start()
                    agent_obs[agent] = normalize_observation(next_obs[agent], observation_tree_depth, observation_radius=observation_radius)
                    preproc_timer.end()

                score += all_rewards[agent]


            if done['__all__']:
                break

        policy.end_episode(True)
        # Epsilon decay
        eps_start = max(eps_end, eps_decay * eps_start)

        # Collect information about training
        tasks_finished = sum([agent.state == TrainState.DONE for agent in train_env.agents])
        completion = tasks_finished / max(1, train_env.get_num_agents())
        normalized_score = score / (max_steps * train_env.get_num_agents())
        # if no actions were ever taken possibly due to malfunction and so 
        # - `actions_taken` is empty [], 
        # - `np.sum(action_count)` is 0
        # Set action probs to count
        if (np.sum(action_count) > 0):
            action_probs = action_count / np.sum(action_count)
        else:
            action_probs = action_count
        action_count = [1] * action_size

        # Set actions_taken to a list with single item 0
        if not actions_taken:
            actions_taken = [0]

        smoothing = 0.99
        smoothed_normalized_score = smoothed_normalized_score * smoothing + normalized_score * (1.0 - smoothing)
        smoothed_completion = smoothed_completion * smoothing + completion * (1.0 - smoothing)

        #own eval while happening
        sparse_scores.append(smoothed_normalized_score)
        sparse_completions.append(smoothed_completion)
        episodes.append(j)
        print(j)

        d.get("networksteps").append(j)
        d.get("score").append(smoothed_normalized_score)
        d.get("algo").append(policy.get_name())

        r.get("networksteps").append(j)
        r.get("completions").append(smoothed_completion)
        r.get("algo").append(policy.get_name() + curriculum)
        if j >= 800000:
            print("networksteps over 2  000 000")
            break

        print(
            '\r🚂 Episode {}'
            '\t 🏆 Score: {:.3f}'
            ' Avg: {:.3f}'
            '\t 💯 Done: {:.2f}%'
            ' Avg: {:.2f}%'
            '\t 🎲 Epsilon: {:.3f} '
            '\t 🔀 Action Probs: {}'.format(
                episode_idx,
                normalized_score,
                smoothed_normalized_score,
                100 * completion,
                100 * smoothed_completion,
                eps_start,
                format_action_prob(action_probs)
            ), end=" ")

    plt.figure(1)
    plt.ylim([-1, 0])
    plt.plot(episodes, sparse_scores)
    plt.savefig('scores.png')

    plt.figure(2)
    plt.ylim([0 , 1])
    plt.plot(episodes, sparse_completions)
    plt.savefig('completions.png')



def format_action_prob(action_probs):
    action_probs = np.round(action_probs, 3)
    actions = ["↻", "←", "↑", "→", "◼"]

    buffer = ""
    for action, action_prob in zip(actions, action_probs):
        buffer += action + " " + "{:.3f}".format(action_prob) + " "

    return buffer




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-n", "--n_episodes", help="number of episodes to run", default= 100000, type=int)
    parser.add_argument("-t", "--training_env_config", help="training config id (eg 0 for Test_0)", default=2, type=int)
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
    parser.add_argument("--use_gpu", help="use GPU if available", default=False, type=bool)
    parser.add_argument("--num_threads", help="number of threads PyTorch can use", default=1, type=int)
    parser.add_argument("--render", help="render 1 episode in 100", default=True, type=bool)
    training_params = parser.parse_args()

    env_params = [
        {
            # Test_0
            "n_agents": 100,
            "x_dim": 256,
            "y_dim": 256,
            "n_cities": 40,
            "max_rails_between_cities": 2,
            "max_rail_pairs_in_city": 4,
            "malfunction_rate": 0,
            "seed": 0
        },
        {
            # Test_1
            "n_agents": 100,
            "x_dim": 128,
            "y_dim": 128,
            "n_cities": 20,
            "max_rails_between_cities": 3,
            "max_rail_pairs_in_city": 4,
            "malfunction_rate": 1 / 1000,
            "seed": 0
        },
        {
            # Test_2
            "n_agents": 20,
            "x_dim": 64,
            "y_dim": 64,
            "n_cities": 10,
            "max_rails_between_cities": 2,
            "max_rail_pairs_in_city": 4,
            "malfunction_rate": 1 / 1000,
            "seed": 0
        },
        {
            # Test_3
            "n_agents": 5,
            "x_dim": 32,
            "y_dim": 32,
            "n_cities": 5,
            "max_rails_between_cities": 3,
            "max_rail_pairs_in_city": 3,
            "malfunction_rate": 1/1000,
            "seed": 0
        },
        {
            # Test_4
            "n_agents": 1,
            "x_dim": 16,
            "y_dim": 18,
            "n_cities": 2,
            "max_rails_between_cities": 1,
            "max_rail_pairs_in_city": 0,
            "malfunction_rate": 0,
            "seed": 0
        },
        {
            # Test für momentane ergebniss 5
            "n_agents": 5, #5
            "x_dim": 30,
            "y_dim": 30,
            "n_cities": 2,
            "max_rails_between_cities": 2,
            "max_rail_pairs_in_city": 2,
            "malfunction_rate": 0,
            "seed": 0
        },
        {
            # Test_6
            "n_agents": 10,
            "x_dim": 30,
            "y_dim": 30,
            "n_cities": 2,
            "max_rails_between_cities": 2,
            "max_rail_pairs_in_city": 2,
            "malfunction_rate": 1 / 100,
            "seed": 0
        },
        {
            # Test_7
            "n_agents": 20,
            "x_dim": 30,
            "y_dim": 30,
            "n_cities": 3,
            "max_rails_between_cities": 2,
            "max_rail_pairs_in_city": 2,
            "malfunction_rate": 0,
            "seed": 0
        },
    ]


    obs_params = {
        "observation_tree_depth": 3,
        "observation_radius": 10,
        "observation_max_path_depth": 30
    }

    def check_env_config(id):
        if id >= len(env_params) or id < 0:
            print("\n🛑 Invalid environment configuration, only Test_0 to Test_{} are supported.".format(len(env_params) - 1))
            exit(1)


    check_env_config(training_params.training_env_config)
    check_env_config(training_params.evaluation_env_config)

    training_env_params = env_params[training_params.training_env_config]
    evaluation_env_params = env_params[training_params.evaluation_env_config]

    print("\nTraining parameters:")
    pprint(vars(training_params))
    print("\nTraining environment parameters (Test_{}):".format(training_params.training_env_config))
    pprint(training_env_params)
    print("\nEvaluation environment parameters (Test_{}):".format(training_params.evaluation_env_config))
    pprint(evaluation_env_params)
    print("\nObservation parameters:")
    pprint(obs_params)

    os.environ["OMP_NUM_THREADS"] = str(training_params.num_threads)
    policies = [DDDQN_UCBPolicy]
    runcounter = 0
    d = {'networksteps': [], 'algo': [], 'score': []}
    r = {'networksteps': [], 'algo': [], 'completions': []}
    for x in policies:
        for y in ["1", "2", "3"]:
            runcounter += 1
            train_agent(training_params, Namespace(**training_env_params), Namespace(**evaluation_env_params), Namespace(**obs_params), x, y, runcounter)



    df = pd.DataFrame(data=d)
    swarm_plot = sns.lineplot(x="networksteps", y="score", hue="algo", data=df)

    swarm_plot.set_ylim(bottom=-1, top=0)
    fig = swarm_plot.get_figure()
    fig.savefig("out.png")


    rf = pd.DataFrame(data=r)
    rswarm_plot = sns.lineplot(x="networksteps", y="completions", hue="algo", data=rf)
    rswarm_plot.set_ylim(bottom=0, top=1)
    rfig = rswarm_plot.get_figure()
    rfig.savefig("out1.png")


