import copy
import os
import pickle
import random
from collections import namedtuple, deque, Iterable
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import copy
import os
from torch.autograd import Variable

import numpy

from reinforcement_learning.model import DuelingQNetwork, DQN, Network
from reinforcement_learning.policy import Policy


class DDDQNPolicy_withewc(Policy):
    """Dueling Double DQN policy"""

    def __init__(self, state_size, action_size, parameters, evaluation_mode=False):
        self.evaluation_mode = evaluation_mode
        self.state_size = state_size
        self.action_size = action_size
        self.double_dqn = True
        self.hidsize = parameters.hidden_size
        self.buffer_size = parameters.buffer_size
        self.batch_size = parameters.batch_size
        self.update_every = parameters.update_every
        self.learning_rate = parameters.learning_rate
        self.tau = parameters.tau
        self.gamma = parameters.gamma
        self.buffer_min_size = parameters.buffer_min_size
        self.evaluation_mode = evaluation_mode
        self.ewc_loss = 0
        self.ewc_lambda = 0.1
        self.retain_graph = False


        # Device
        if parameters.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:2")
            print("🐇 Using GPU")
        else:
            self.device = torch.device("cpu")
            print("🐢 Using CPU")

        # Q-Network
        self.qnetwork_local = DuelingQNetwork(state_size, action_size, hidsize1=self.hidsize, hidsize2=self.hidsize).to(self.device)
        self.qnetwork_target = copy.deepcopy(self.qnetwork_local)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, self.device)

        self.t_step = 0
        self.loss = 0.0
        self.params = {n: p for n, p in self.qnetwork_local.named_parameters() if p.requires_grad}
        self.p_old = {}

        for n, p in copy.deepcopy(self.params).items():
            self.p_old[n] = torch.Tensor(p.data)

    def act(self,handle, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, handle, state, action, reward, next_state, done):
        assert not self.evaluation_mode, "Policy has been initialized for evaluation only."

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.buffer_min_size and len(self.memory) > self.batch_size:
                self._learn()

    def _learn(self):
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences

        # Get expected Q values from local model
        q_expected = self.qnetwork_local(states).gather(1, actions)

        if self.double_dqn:
            # Double DQN
            q_best_action = self.qnetwork_local(next_states).max(1)[1]
            q_targets_next = self.qnetwork_target(next_states).gather(1, q_best_action.unsqueeze(-1))
        else:
            # DQN
            q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(-1)

        # Compute Q targets for current states
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))


        # Compute the total loss
        self.loss = F.mse_loss(q_expected, q_targets) + self.ewc_lambda * self.ewc_loss
        # Minimize the loss
        self.optimizer.zero_grad()
        self.loss.backward(retain_graph=self.retain_graph)

        self.optimizer.step()

        # Update target network
        self._soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)




    def get_ewc_loss(self, model, fisher, p_old):
        #named parameters checken, da ist irgendwo nen fehler, dass die niocht gleichbenannet sind
        loss = 0
        for n, p in model.named_parameters():
            _loss = fisher[n] * (p - p_old[n]) ** 2
            loss += _loss.sum()
        return loss
    def update_ewc(self):
        fisher_matrix = self.get_fisher_diag(self.qnetwork_local, self.memory, self.params)
        self.ewc_loss += self.get_ewc_loss(self.qnetwork_local, fisher_matrix, self.p_old)

        self.retain_graph = True
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.device)
        self.params = {n: p for n, p in self.qnetwork_local.named_parameters() if p.requires_grad}
        self.p_old ={}
        for n, p in copy.deepcopy(self.params).items():
            self.p_old[n] = torch.Tensor(p.data)

    def get_fisher_diag(self, model, dataset, params):
        fisher = {}
        for n, p in copy.deepcopy(params).items():
            p.data.zero_()
            fisher[n] = Variable(p.data)

        model.eval()
        states, actions, rewards, next_states, done = dataset.sample()
        #
        for x in range(states.size(dim = 0)):
            model.zero_grad()
            output = model(states[x]).view(1, -1)
            target = output.max(1)[1].view(-1)
            negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), target)
            negloglikelihood.backward(retain_graph=True)
        #
        for n, p in model.named_parameters():
            fisher[n].data += p.grad.data ** 2 / 128

        fisher = {n: p for n, p in fisher.items()}
        return fisher





    def _soft_update(self, local_model, target_model, tau):
        # Soft update model parameters.
        # θ_target = τ*θ_local + (1 - τ)*θ_target
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)




Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
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

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = Experience(np.expand_dims(state, 0), action, reward, np.expand_dims(next_state, 0), done)
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

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def __v_stack_impr(self, states):
        sub_dim = len(states[0][0]) if isinstance(states[0], Iterable) else 1
        np_states = np.reshape(np.array(states), (len(states), sub_dim))
        return np_states
