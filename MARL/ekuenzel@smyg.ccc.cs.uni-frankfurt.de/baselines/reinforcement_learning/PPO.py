import torch.optim as optim
from torch.distributions import Categorical


import random
from collections import namedtuple, deque
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

from collections import namedtuple
from typing import Union

import torch
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import copy
import os

import numpy as np
import torch
import torch.nn as nn

from reinforcement_learning.policy import LearningPolicy



# https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html


class EpisodeBuffers:
    def __init__(self):
        self.reset()

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def reset(self):
        self.memory = {}

    def get_transitions(self, handle):
        return self.memory.get(handle, [])

    def push_transition(self, handle, transition):
        transitions = self.get_transitions(handle)
        transitions.append(transition)
        self.memory.update({handle: transitions})


# Actor module
class FeatureExtractorNetwork(nn.Module):
    def __init__(self, state_size, device, hidsize1=512, hidsize2=256):
        super(FeatureExtractorNetwork, self).__init__()
        self.device = device
        self.nn_layer_outputsize = hidsize2
        self.model = nn.Sequential(
            nn.Linear(state_size, hidsize1),
            nn.Tanh(),
            nn.Linear(hidsize1, hidsize2),
            nn.Tanh()
        ).to(self.device)

    def forward(self, X):
        return self.model(X)

    def save(self, filename):
        # print("Saving model from checkpoint:", filename)
        torch.save(self.model.state_dict(), filename + ".ppo_feature_extractor")

    def _load(self, obj, filename):
        if os.path.exists(filename):
            print(' >> ', filename)
            try:
                obj.load_state_dict(torch.load(filename, map_location=self.device))
            except:
                print(" >> failed!")
        return obj

    def load(self, filename):
        print("load model from file", filename)
        self.model = self._load(self.model, filename + ".ppo_feature_extractor")
class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, device, feature_extractor_model: FeatureExtractorNetwork = None,
                 hidsize=256,
                 learning_rate=0.5e-3):
        super(ActorNetwork, self).__init__()
        self.device = device
        self.feature_extractor_model = feature_extractor_model
        self.model = nn.Sequential(
            nn.Linear(state_size, hidsize) if (self.feature_extractor_model is None)
            else nn.Linear(feature_extractor_model.nn_layer_outputsize, hidsize),
            nn.Tanh(),
            nn.Linear(hidsize, hidsize),
            nn.Tanh(),
            nn.Linear(hidsize, action_size),
            nn.Softmax(dim=-1)
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, input):
        if self.feature_extractor_model is None:
            return self.model(input)
        return self.model(self.feature_extractor_model(input))

    def get_actor_dist(self, state):
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs=probs)
        return dist, probs

    def evaluate(self, states, actions):
        dist, action_probs = self.get_actor_dist(states)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        return action_logprobs, dist_entropy

    def save(self, filename):
        torch.save(self.model.state_dict(), filename + ".ppo_actor")
        torch.save(self.optimizer.state_dict(), filename + ".ppo_optimizer_actor")

    def _load(self, obj, filename):
        if os.path.exists(filename):
            print(' >> ', filename)
            try:
                obj.load_state_dict(torch.load(filename, map_location=self.device))
            except:
                print(" >> failed!")
        return obj

    def load(self, filename):
        print("load model from file", filename)
        self.model = self._load(self.model, filename + ".ppo_actor")
        self.optimizer = self._load(self.optimizer, filename + ".ppo_optimizer_actor")


# Critic module
class CriticNetwork(nn.Module):
    def __init__(self, state_size, device, feature_extractor_model: FeatureExtractorNetwork = None, hidsize=256,
                 learning_rate=0.5e-3):
        super(CriticNetwork, self).__init__()
        self.device = device
        self.feature_extractor_model = feature_extractor_model
        self.model = nn.Sequential(
            nn.Linear(state_size, hidsize) if (self.feature_extractor_model is None)
            else nn.Linear(feature_extractor_model.nn_layer_outputsize, hidsize),
            nn.Tanh(),
            nn.Linear(hidsize, 1)
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, input):
        if self.feature_extractor_model is None:
            return self.model(input)
        return self.model(self.feature_extractor_model(input))

    def evaluate(self, states):
        state_value = self.forward(states)
        return torch.squeeze(state_value)

    def save(self, filename):
        torch.save(self.model.state_dict(), filename + ".ppo_critic")
        torch.save(self.optimizer.state_dict(), filename + ".ppo_optimizer_critic")

    def _load(self, obj, filename):
        if os.path.exists(filename):
            print(' >> ', filename)
            try:
                obj.load_state_dict(torch.load(filename, map_location=self.device))
            except:
                print(" >> failed!")
        return obj

    def load(self, filename):
        print("load model from file", filename)
        self.model = self._load(self.model, filename + ".ppo_critic")
        self.optimizer = self._load(self.optimizer, filename + ".ppo_optimizer_critic")




Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "action_prob"])
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device

    def add(self, state, action, reward, next_state, done, action_prob=0.0):
        """Add a new experience to memory."""
        e = Experience(np.expand_dims(state, 0), action, reward, np.expand_dims(next_state, 0), done, action_prob)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(self.__v_stack_impr([e.state for e in experiences if e is not None])) \
            .float().to(self.device)
        actions = torch.from_numpy(self.__v_stack_impr([e.action for e in experiences if e is not None])) \
            .long().to(self.device)
        rewards = torch.from_numpy(self.__v_stack_impr([e.reward for e in experiences if e is not None])) \
            .float().to(self.device)
        next_states = torch.from_numpy(self.__v_stack_impr([e.next_state for e in experiences if e is not None])) \
            .float().to(self.device)
        dones = torch.from_numpy(self.__v_stack_impr([e.done for e in experiences if e is not None]).astype(np.uint8)) \
            .float().to(self.device)
        action_probs = torch.from_numpy(self.__v_stack_impr([e.action_prob for e in experiences if e is not None])) \
            .float().to(self.device)

        return states, actions, rewards, next_states, dones, action_probs

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def __v_stack_impr(self, states):
        sub_dim = len(states[0][0]) if isinstance(states[0], Iterable) else 1
        np_states = np.reshape(np.array(states), (len(states), sub_dim))
        return np_states
# https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html

# Proximal Policy Optimization (PPO)
# https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html

class EpisodeBuffers:
    def __init__(self):
        self.reset()

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def reset(self):
        self.memory = {}

    def get_transitions(self, handle):
        return self.memory.get(handle, [])

    def push_transition(self, handle, transition):
        transitions = self.get_transitions(handle)
        transitions.append(transition)
        self.memory.update({handle: transitions})



#PPO
class ActorCriticModel(nn.Module):

    def __init__(self, state_size, action_size, device, hidsize1=512, hidsize2=256):
        super(ActorCriticModel, self).__init__()
        self.device = device
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidsize1),
            nn.Tanh(),
            nn.Linear(hidsize1, hidsize2),
            nn.Tanh(),
            nn.Linear(hidsize2, action_size),
            nn.Softmax(dim=-1)
        ).to(self.device)

        self.critic = nn.Sequential(
            nn.Linear(state_size, hidsize1),
            nn.Tanh(),
            nn.Linear(hidsize1, hidsize2),
            nn.Tanh(),
            nn.Linear(hidsize2, 1)
        ).to(self.device)

    def forward(self, x):
        raise NotImplementedError

    def get_actor_dist(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        return dist

    def evaluate(self, states, actions):
        action_probs = self.actor(states)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_value = self.critic(states)
        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def save(self, filename):
        # print("Saving model from checkpoint:", filename)
        torch.save(self.actor.state_dict(), filename + ".actor")
        torch.save(self.critic.state_dict(), filename + ".value")

    def _load(self, obj, filename):
        if os.path.exists(filename):
            print(' >> ', filename)
            try:
                obj.load_state_dict(torch.load(filename, map_location=self.device))
            except:
                print(" >> failed!")
        return obj

    def load(self, filename):
        print("load model from file", filename)
        self.actor = self._load(self.actor, filename + ".actor")
        self.critic = self._load(self.critic, filename + ".value")
PPO_Param = namedtuple('PPO_Param',['hidden_size', 'buffer_size', 'batch_size', 'learning_rate', 'discount', 'buffer_min_size', 'use_replay_buffer', 'use_gpu'])
class PPOPolicy(LearningPolicy):
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 parameters):
        super(PPOPolicy, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = parameters.hidden_size
        self.buffer_size = parameters.buffer_size
        self.batch_size = parameters.batch_size
        self.update_every = parameters.update_every
        self.learning_rate = parameters.learning_rate
        self.tau = parameters.tau
        self.gamma = parameters.gamma
        self.buffer_min_size = parameters.buffer_min_size
        self.discount = 0.95
        self.use_replay_buffer = True
        self.device = torch.device("cpu")

        # Device
        if parameters.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("🐇 Using GPU")
        else:
            self.device = torch.device("cpu")
            print("🐢 Using CPU")

        self.surrogate_eps_clip = 0.1
        self.K_epoch = 10
        self.weight_loss = 0.5
        self.weight_entropy = 0.01

        self.buffer_min_size = 0

        self.current_episode_memory = EpisodeBuffers()
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, self.device)
        self.loss = 0
        self.actor_critic_model = ActorCriticModel(state_size, action_size, self.device,
                                                   hidsize1=self.hidden_size,
                                                   hidsize2=self.hidden_size)
        self.optimizer = optim.Adam(self.actor_critic_model.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()  # nn.SmoothL1Loss()

    def get_name(self):
        return self.__class__.__name__

    def act(self, handle, state, eps=None):
        # sample a action to take
        torch_state = torch.tensor(state, dtype=torch.float).to(self.device)
        dist = self.actor_critic_model.get_actor_dist(torch_state)
        action = dist.sample()
        return action.item()

    def step(self, handle, state, action, reward, next_state, done):
        # record transitions ([state] -> [action] -> [reward, next_state, done])
        torch_action = torch.tensor(action, dtype=torch.float).to(self.device)
        torch_state = torch.tensor(state, dtype=torch.float).to(self.device)
        # evaluate actor
        dist = self.actor_critic_model.get_actor_dist(torch_state)
        action_logprobs = dist.log_prob(torch_action)
        transition = (state, action, reward, next_state, action_logprobs.item(), done)
        self.current_episode_memory.push_transition(handle, transition)

    def _push_transitions_to_replay_buffer(self,
                                           state_list,
                                           action_list,
                                           reward_list,
                                           state_next_list,
                                           done_list,
                                           prob_a_list):
        for idx in range(len(reward_list)):
            state_i = state_list[idx]
            action_i = action_list[idx]
            reward_i = reward_list[idx]
            state_next_i = state_next_list[idx]
            done_i = done_list[idx]
            prob_action_i = prob_a_list[idx]
            self.memory.add(state_i, action_i, reward_i, state_next_i, done_i, prob_action_i)

    def _convert_transitions_to_torch_tensors(self, transitions_array):
        # build empty lists(arrays)
        state_list, action_list, reward_list, state_next_list, prob_a_list, done_list = [], [], [], [], [], []

        # set discounted_reward to zero
        discounted_reward = 0
        for transition in transitions_array[::-1]:
            state_i, action_i, reward_i, state_next_i, prob_action_i, done_i = transition

            state_list.insert(0, state_i)
            action_list.insert(0, action_i)
            if done_i:
                discounted_reward = 0
                done_list.insert(0, 1)
            else:
                done_list.insert(0, 0)

            discounted_reward = reward_i + self.discount * discounted_reward
            reward_list.insert(0, discounted_reward)
            state_next_list.insert(0, state_next_i)
            prob_a_list.insert(0, prob_action_i)

        if self.use_replay_buffer:
            self._push_transitions_to_replay_buffer(state_list, action_list,
                                                    reward_list, state_next_list,
                                                    done_list, prob_a_list)

        # convert data to torch tensors
        states, actions, rewards, states_next, dones, prob_actions = \
            torch.tensor(state_list, dtype=torch.float).to(self.device), \
                torch.tensor(action_list).to(self.device), \
                torch.tensor(reward_list, dtype=torch.float).to(self.device), \
                torch.tensor(state_next_list, dtype=torch.float).to(self.device), \
                torch.tensor(done_list, dtype=torch.float).to(self.device), \
                torch.tensor(prob_a_list).to(self.device)

        return states, actions, rewards, states_next, dones, prob_actions

    def _get_transitions_from_replay_buffer(self, states, actions, rewards, states_next, dones, probs_action):
        if len(self.memory) > self.buffer_min_size and len(self.memory) > self.batch_size:
            states, actions, rewards, states_next, dones, probs_action = self.memory.sample()
            actions = torch.squeeze(actions)
            rewards = torch.squeeze(rewards)
            states_next = torch.squeeze(states_next)
            dones = torch.squeeze(dones)
            probs_action = torch.squeeze(probs_action)
        return states, actions, rewards, states_next, dones, probs_action

    def train_net(self):
        # All agents have to propagate their experiences made during past episode
        for handle in range(len(self.current_episode_memory)):
            # Extract agent's episode history (list of all transitions)
            agent_episode_history = self.current_episode_memory.get_transitions(handle)
            if len(agent_episode_history) > 0:
                # Convert the replay buffer to torch tensors (arrays)
                states, actions, rewards, states_next, dones, probs_action = \
                    self._convert_transitions_to_torch_tensors(agent_episode_history)

                # Optimize policy for K epochs:
                for k_loop in range(int(self.K_epoch)):

                    if self.use_replay_buffer:
                        states, actions, rewards, states_next, dones, probs_action = \
                            self._get_transitions_from_replay_buffer(
                                states, actions, rewards, states_next, dones, probs_action
                            )

                    # Evaluating actions (actor) and values (critic)
                    logprobs, state_values, dist_entropy = self.actor_critic_model.evaluate(states, actions)

                    # Finding the ratios (pi_thetas / pi_thetas_replayed):
                    ratios = torch.exp(logprobs - probs_action.detach())

                    # Finding Surrogate Loos
                    advantages = rewards - state_values.detach()
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1. - self.surrogate_eps_clip, 1. + self.surrogate_eps_clip) * advantages

                    # The loss function is used to estimate the gardient and use the entropy function based
                    # heuristic to penalize the gradient function when the policy becomes deterministic this would let
                    # the gradient becomes very flat and so the gradient is no longer useful.
                    loss = \
                        -torch.min(surr1, surr2) \
                        + self.weight_loss * self.loss_function(state_values, rewards) \
                        - self.weight_entropy * dist_entropy

                    # Make a gradient step
                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    self.optimizer.step()

                    # Transfer the current loss to the agents loss (information) for debug purpose only
                    self.loss = loss.mean().detach().cpu().numpy()

        # Reset all collect transition data
        self.current_episode_memory.reset()

    def end_episode(self, train):
        if train:
            self.train_net()

    def get_name(self):
        return "PPO"





# ActorCritic
class A2CShared(nn.Module):
    def __init__(self, state_size, action_size, device, hidsize1=512, hidsize2=256):
        super(A2CShared, self).__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(state_size, hidsize1),
            nn.Tanh(),
            nn.Linear(hidsize1, hidsize2),
            nn.Tanh(),
            nn.Linear(hidsize2, hidsize2),
            nn.Tanh()
        ).to(self.device)

    def forward(self, X):
        return self.model(X)

    def save(self, filename):
        # print("Saving model from checkpoint:", filename)
        torch.save(self.model.state_dict(), filename + ".a2c_shared")

    def _load(self, obj, filename):
        if os.path.exists(filename):
            print(' >> ', filename)
            try:
                obj.load_state_dict(torch.load(filename, map_location=self.device))
            except:
                print(" >> failed!")
        return obj

    def load(self, filename):
        print("load model from file", filename)
        self.model = self._load(self.model, filename + ".a2c_shared")
class A2CActor(nn.Module):
    def __init__(self, state_size, action_size, device, shared_model, hidsize1=512, hidsize2=256):
        super(A2CActor, self).__init__()
        self.device = device
        self.shared_model = shared_model
        self.model = nn.Sequential(
            nn.Linear(hidsize2, hidsize2),
            nn.Tanh(),
            nn.Linear(hidsize2, action_size),
            nn.Softmax(dim=-1)
        ).to(self.device)

    def forward(self, X):
        return self.model(self.shared_model(X))

    def get_actor_dist(self, state):
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs=probs)
        return dist, probs

    def save(self, filename):
        # print("Saving model from checkpoint:", filename)
        torch.save(self.model.state_dict(), filename + ".a2c_actor")

    def _load(self, obj, filename):
        if os.path.exists(filename):
            print(' >> ', filename)
            try:
                obj.load_state_dict(torch.load(filename, map_location=self.device))
            except:
                print(" >> failed!")
        return obj

    def load(self, filename):
        print("load model from file", filename)
        self.model = self._load(self.model, filename + ".a2c_actor")
# Critic module
class A2CCritic(nn.Module):
    def __init__(self, state_size, device, shared_model, hidsize1=512, hidsize2=256):
        super(A2CCritic, self).__init__()
        self.device = device
        self.shared_model = shared_model
        self.model = nn.Sequential(
            nn.Linear(hidsize2, hidsize2),
            nn.Tanh(),
            nn.Linear(hidsize2, 1)
        ).to(self.device)

    def forward(self, X):
        return self.model(self.shared_model(X))

    def save(self, filename):
        # print("Saving model from checkpoint:", filename)
        torch.save(self.model.state_dict(), filename + ".a2c_critic")

    def _load(self, obj, filename):
        if os.path.exists(filename):
            print(' >> ', filename)
            try:
                obj.load_state_dict(torch.load(filename, map_location=self.device))
            except:
                print(" >> failed!")
        return obj

    def load(self, filename):
        print("load model from file", filename)
        self.model = self._load(self.model, filename + ".a2c_critic")
class A2CPolicy(LearningPolicy):
    def __init__(self, state_size, action_size, in_parameters=None, clip_grad_norm=0.1):
        print(">> A2CPolicy")
        super(A2CPolicy, self).__init__()
        # parameters
        self.state_size = state_size
        self.action_size = action_size
        self.a2c_parameters = in_parameters
        if self.a2c_parameters is not None:
            self.hidsize = self.a2c_parameters.hidden_size
            self.learning_rate = self.a2c_parameters.learning_rate
            self.gamma = self.a2c_parameters.gamma
            # Device
            if self.a2c_parameters.use_gpu and torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                print("🐇 Using GPU")
            else:
                self.device = torch.device("cpu")
                print("🐢 Using CPU")
        else:
            self.hidsize = 128
            self.learning_rate = 0.5e-4
            self.gamma = 0.99
            self.device = torch.device("cpu")

        self.current_episode_memory = EpisodeBuffers()

        self.agent_done = {}
        self.memory = []  # dummy parameter

        self.loss_function = nn.MSELoss()
        self.loss = 0
        self.update_every = 8
        self.t_step = 0

        self.shared_model = A2CShared(state_size,
                                      action_size,
                                      self.device,
                                      hidsize1=self.hidsize,
                                      hidsize2=self.hidsize)

        self.actor = A2CActor(state_size,
                              action_size,
                              self.device,
                              shared_model=self.shared_model,
                              hidsize1=self.hidsize,
                              hidsize2=self.hidsize)
        self.critic = A2CCritic(state_size,
                                self.device,
                                shared_model=self.shared_model,
                                hidsize1=self.hidsize,
                                hidsize2=self.hidsize)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        self.clip_grad_norm = clip_grad_norm

    def shape_reward(self, handle, action, state, reward, done, deadlocked=None):
        if done == 1:
            return 1.0
        return 0.0

    def reset(self, env):
        pass

    def act(self, handle, state, eps=0.0):
        torch_state = torch.tensor(state, dtype=torch.float).to(self.device)
        dist, _ = self.actor.get_actor_dist(torch_state)
        action = dist.sample()
        return action.item()

    def step(self, handle, state, action, reward, next_state, done):

        if self.agent_done.get(handle, False):
            return  # remove? if not Flatland?
        # record transitions ([state] -> [action] -> [reward, next_state, done])
        torch_action = torch.tensor(action, dtype=torch.float).to(self.device)
        torch_state = torch.tensor(state, dtype=torch.float).to(self.device)
        # evaluate actor
        dist, _ = self.actor.get_actor_dist(torch_state)
        action_logprobs = dist.log_prob(torch_action)
        transition = (state, action, reward, next_state, action_logprobs.item(), done)
        self.current_episode_memory.push_transition(handle, transition)
        if done:
            self.agent_done.update({handle: done})





    def _convert_transitions_to_torch_tensors(self, transitions_array, all_done):
        # build empty lists(arrays)
        state_list, action_list, reward_list, state_next_list, prob_a_list, done_list = [], [], [], [], [], []

        # set discounted_reward to zero
        discounted_reward = 0
        for transition in transitions_array[::-1]:
            state_i, action_i, reward_i, state_next_i, prob_action_i, done_i = transition
            state_list.insert(0, state_i)
            action_list.insert(0, action_i)
            done_list.insert(0, int(done_i))
            mask_i = 1.0 - int(done_i)
            discounted_reward = reward_i + self.gamma * mask_i * discounted_reward
            reward_list.insert(0, discounted_reward)
            state_next_list.insert(0, state_next_i)
            prob_a_list.insert(0, prob_action_i)

        # convert data to torch tensors
        states, actions, rewards, states_next, dones, prob_actions = \
            torch.tensor(state_list, dtype=torch.float).to(self.device), \
            torch.tensor(action_list).to(self.device), \
            torch.tensor(reward_list, dtype=torch.float).to(self.device), \
            torch.tensor(state_next_list, dtype=torch.float).to(self.device), \
            torch.tensor(done_list, dtype=torch.float).to(self.device), \
            torch.tensor(prob_a_list).to(self.device)

        return states, actions, rewards, states_next, dones, prob_actions

    def train_net(self):
        # All agents have to propagate their experiences made during past episode
        all_done = True
        for handle in range(len(self.current_episode_memory)):
            all_done = self.agent_done.get(handle, False)

        for handle in range(len(self.current_episode_memory)):
            # Extract agent's episode history (list of all transitions)
            agent_episode_history = self.current_episode_memory.get_transitions(handle)
            if len(agent_episode_history) > 0:
                # Convert the replay buffer to torch tensors (arrays)
                states, actions, rewards, states_next, dones, probs_action = \
                    self._convert_transitions_to_torch_tensors(agent_episode_history, all_done)

                critic_loss = self.loss_function(rewards, torch.squeeze(self.critic(states)))
                self.optimizer_critic.zero_grad()
                critic_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.critic.model.parameters(), self.clip_grad_norm)
                self.optimizer_critic.step()


                advantages = rewards - torch.squeeze(self.critic(states)).detach()



                dist, _ = self.actor.get_actor_dist(states)
                action_logprobs = dist.log_prob(actions)
                self.optimizer_actor.zero_grad()
                actor_loss = -(advantages.detach() * action_logprobs) - 0.01 * dist.entropy()
                actor_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor.model.parameters(), self.clip_grad_norm)
                self.optimizer_actor.step()

                # Transfer the current loss to the agents loss (information) for debug purpose only
                self.loss = actor_loss.mean().detach().cpu().numpy()
        # Reset all collect transition data
        self.current_episode_memory.reset()
        self.agent_done = {}

    def end_episode(self, train):
        if train:
            self.train_net()
    def get_name(self):
        return "A2C"





# Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidsize=256):
        super(Actor, self).__init__()
        self.device = device
        self.state_size = state_size
        self.action_size = action_size

        self.model = nn.Sequential(
            nn.Linear(state_size, hidsize),
            nn.Tanh(),
            nn.Linear(hidsize, hidsize),
            nn.Tanh(),
            nn.Linear(hidsize, action_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.model(state)
class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidsize=256):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        # Q1 architecture
        self.model_q1 = nn.Sequential(
            nn.Linear(state_size + action_size, hidsize),
            nn.LeakyReLU(),
            nn.Linear(hidsize, hidsize),
            nn.LeakyReLU(),
            nn.Linear(hidsize, 1),
            nn.LeakyReLU()
        )

        # Q2 architecture
        self.model_q2 = nn.Sequential(
            nn.Linear(state_size + action_size, hidsize),
            nn.LeakyReLU(),
            nn.Linear(hidsize, hidsize),
            nn.LeakyReLU(),
            nn.Linear(hidsize, 1),
            nn.LeakyReLU()
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.model_q1(sa)
        q2 = self.model_q2(sa)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.model_q1(sa)
T3D_Param = namedtuple('T3D_Param',
                       ['hidden_size',
                        'buffer_size',
                        'buffer_min_size',
                        'batch_size',
                        'discount',
                        'learning_rate',
                        'tau',
                        'policy_noise',
                        'noise_clip',
                        'policy_freq',
                        'use_gpu'])
class TD3Policy(LearningPolicy):
    def __init__(
            self,
            state_size,
            action_size,
            in_parameters: Union[T3D_Param, None] = None
    ):
        super(TD3Policy, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        self.t3d_param = in_parameters

        if False:
            self.buffer_size = self.t3d_param.buffer_size
            self.batch_size = self.t3d_param.batch_size
            self.buffer_min_size = self.t3d_param.buffer_min_size
            self.learning_rate = self.t3d_param.learning_rate
            self.discount = self.t3d_param.discount
            self.tau = self.t3d_param.tau
            self.policy_noise = self.t3d_param.policy_noise
            self.noise_clip = self.t3d_param.noise_clip
            self.policy_freq = self.t3d_param.policy_freq
        else:
            self.buffer_size = 32_000
            self.batch_size = 256
            self.buffer_min_size = 0
            self.learning_rate = 0.5e-4
            self.discount = 0.95
            self.tau = 0.1e-2
            self.policy_noise = 0.2
            self.noise_clip = 0.1
            self.policy_freq = 2

        # Device
        if self.t3d_param.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("🐇 Using GPU")
        else:
            self.device = torch.device("cpu")
            print("🐢 Using CPU")

        self.actor = Actor(state_size, action_size).to(self.device)
        self.actor_target = Actor(state_size, action_size).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)

        self.critic = Critic(state_size, action_size).to(self.device)
        self.critic_target = Critic(state_size, action_size).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, self.device)
        self.loss = 0
        self.total_it = 0

        self.t_step = 0
        self.update_every = 5

    def get_name(self):
        return self.__class__.__name__

    def act(self, handle, state, eps=0.):
        # Epsilon-greedy action selection
        if np.random.random() > eps:
            state = torch.FloatTensor(np.array(state).reshape(1, -1)).to(self.device)
            estimated = self.actor(state)
            return np.argmax(estimated.cpu().data.numpy())
        else:
            return np.random.choice(np.arange(self.action_size))

    def step(self, handle, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            self._train_net()

    def _train_net(self):
        if len(self.memory) <= self.buffer_min_size or len(self.memory) <= self.batch_size:
            return

        self.total_it += 1

        # Sample replay buffer
        states, actions, rewards, states_next, dones, _ = self.memory.sample()
        actions = F.one_hot(torch.squeeze(actions), num_classes=self.action_size)
        states_next = torch.squeeze(states_next)
        rewards = torch.squeeze(rewards)
        dones = torch.squeeze(dones)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn(states_next.shape[0], states_next.shape[1]) * self.policy_noise) \
                .clamp(-self.noise_clip, self.noise_clip) \
                .to(self.device)
            noisy_actions = self.actor_target(states_next + noise)
            noisy_action_greedy = torch.argmax(noisy_actions, dim=1)
            next_actions = F.one_hot(noisy_action_greedy, num_classes=self.action_size)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(states_next, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = torch.reshape(rewards, (len(rewards), 1)) + \
                       torch.reshape(1.0 - dones, (len(dones), 1)) * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(states, actions)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
                      F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            self._soft_update(self.critic, self.critic_target, self.tau)
            self._soft_update(self.actor, self.actor_target, self.tau)

        self.loss = critic_loss.mean().detach().cpu().numpy()

    def _soft_update(self, local_model, target_model, tau):
        # Soft update model parameters.
        # θ_target = τ*θ_local + (1 - τ)*θ_target
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
        print('{} -> load {} ok'.format(self.get_name(), filename))

    def clone(self):
        policy = TD3Policy(self.state_size, self.action_size, self.t3d_param)
        policy.actor = copy.deepcopy(self.actor)
        policy.critic = copy.deepcopy(self.critic)
        policy.actor_optimizer = copy.deepcopy(self.actor_optimizer)
        policy.critic_optimizer = copy.deepcopy(self.critic_optimizer)
        return policy
    def end_episode(self, train):
        pass
    def get_name(self):
        return "TD3"