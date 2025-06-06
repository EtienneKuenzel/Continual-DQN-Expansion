import pickle
from collections import namedtuple, deque, Iterable
from torch.autograd import Variable
import pandas as pd
import heapq
import numpy
import torch
import torch.optim as optim
import copy
import random
import numpy as np
from collections import defaultdict
from reinforcement_learning.model import DQN, Network
torch.set_printoptions(precision=5)
import numpy as np
import torch
from torch import nn
from reinforcement_learning.gnt import GnT
import torch.nn.functional as F


class Continual_DQN_Expansionold():
    """Continual DDQN Expansion(CDE)"""
    def __init__(self, state_size, action_size, parameters, evaluation_mode=False):
        self.networks = [[]]
        self.state_size = state_size
        self.action_size = action_size
        self.parameters = parameters
        self.evaluation_mode = evaluation_mode
        self.networks[0].append(subCDE_Policy(state_size, action_size, parameters, evaluation_mode))
        self.anchor_number = self.parameters.anchors
        self.act_rotation = 0
        self.evaluation_mode = False
        self.networkEP = []
        self.networkEP_scores = []
        self.networkEP_completions = []
    def act(self, handle, state, eps=0., eval = False):
        if eval:
            best_average = -1
            best_network = self.networks[-1][0]
            for network in self.networks[-1]:
                average = sum(network.score) / len(network.score)
                if average > best_average:
                    best_network = network
                    best_average = average
            return best_network.act(handle, state, eps)
        else:
            return self.networks[-1][self.act_rotation].act(handle, state, eps)
    def network_rotation(self, score, completions):
        self.networks[-1][self.act_rotation].score.pop(0)  # Remove the oldest element
        self.networks[-1][self.act_rotation].score.append(score)
        self.networks[-1][self.act_rotation].completions.pop(0)  # Remove the oldest element
        self.networks[-1][self.act_rotation].completions.append(completions)
        self.act_rotation +=1
        self.act_rotation %= len(self.networks[-1])
    def step(self, handle, state, action, reward, next_state, done):
        for network in self.networks[-1]:
            network.step(handle, state, action, reward, next_state, done)
    def expansion(self):
        last_networks = self.networks[-1]
        self.act_rotation = 0
        # Use nlargest to keep the top 2 elements with the highest average score, including their indexes.
        top_networks = heapq.nlargest(self.anchor_number, [(numpy.average(x.score), i, x) for i, x in enumerate(last_networks)], key=lambda item: item[0])
        # Extract the elements (discard the average scores) and indexes, then update self.networks[-1].
        top_network_indexes = [i for _, i, _ in top_networks]
        self.networks[-1] = [x for _, _, x in top_networks]
        self.networkEP.append(top_network_indexes)
        self.networkEP_scores.append([numpy.mean(last_networks[x].score) for x in top_network_indexes])
        self.networkEP_completions.append([numpy.mean(last_networks[x].completions) for x in top_network_indexes])
        #double the amount of networks first half= EWC last half =PAU
        a = len(self.networks[-1])
        for network in range(a):
            self.networks[-1].append(subCDE_Policy(self.state_size, self.action_size, self.parameters, self.evaluation_mode, freeze=False, initialweights=self.networks[-1][network].get_weigths()))
            self.networks[-1][-1].set_parameters(self.networks[-1][network].qnetwork_local,
                                                self.networks[-1][network].qnetwork_target,
                                                self.networks[-1][network].optimizer, self.networks[-1][network].ewc_loss, self.networks[-1][network].memory,
                                                self.networks[-1][network].loss, self.networks[-1][network].params,
                                                self.networks[-1][network].p_old)
            self.networks[-1][network].update_ewc()
            self.networks[-1][network].freeze = True

    def set_evaluation_mode(self, evaluation_mode):
        self.evaluation_mode = evaluation_mode
    def get_name(self):
        return "CDE" + str(self.act_rotation)
    def get_activation(self):
        result = []
        for param_tuple in self.networks[-1][self.act_rotation].get_weigths():
            a, b = param_tuple
            result.append([a.tolist(), b.tolist()])
        return result
    def get_net(self):
        return self.act_rotation
    def get_fisher(self):
        return self.get_fisher_diag(self.qnetwork_local, self.memory, self.params)
class Continual_DQN_Expansion():
    """Continual DDQN Expansion(CDE)"""
    def __init__(self, state_size, action_size, parameters, evaluation_mode=False):
        self.networks = [[]]
        self.state_size = state_size
        self.action_size = action_size
        self.parameters = parameters
        self.evaluation_mode = evaluation_mode
        self.networks[0].append(subCDE_Policy(state_size, action_size, parameters, evaluation_mode))
        self.anchor_number = self.parameters.anchors
        self.act_rotation = 0
        self.evaluation_mode = False
        self.networkEP = []
        self.networkEP_scores = []
        self.networkEP_completions = []
        self.a = 0


    def act(self, handle, state, eps=0., eval=False, besteval=True):
        if eval:
            if besteval:
                return self.networks[-1][self.act_rotation].act(handle, state, eps)
            else:
                actions = [network.act(handle, state, eps) for network in self.networks[-1]]
                counts = np.bincount(actions)
                top_actions = np.where(counts == counts.max())[0]
                return np.random.choice(top_actions)
        else:
            return self.networks[-1][self.act_rotation].act(handle, state, eps)

    def network_rotation(self, score, completions):
        self.networks[-1][self.act_rotation].score.pop(0)  # Remove the oldest element
        self.networks[-1][self.act_rotation].score.append(score)
        self.networks[-1][self.act_rotation].completions.pop(0)  # Remove the oldest element
        self.networks[-1][self.act_rotation].completions.append(completions)
        self.act_rotation +=1
        self.act_rotation %= len(self.networks[-1])
    def step(self, handle, state, action, reward, next_state, done):
        for network in self.networks[-1]:
            network.step(handle, state, action, reward, next_state, done)
    def pruning(self):
        last_networks = self.networks[-1]
        self.act_rotation = 0
        # Use nlargest to keep the top 2 elements with the highest average score, including their indexes.
        top_networks = heapq.nlargest(self.anchor_number,[(numpy.average(x.score), i, x) for i, x in enumerate(last_networks)],key=lambda item: item[0])
        # Extract the elements (discard the average scores) and indexes, then update self.networks[-1].
        top_network_indexes = [i for _, i, _ in top_networks]
        self.networks[-1] = [x for _, _, x in top_networks]
        self.networkEP.append(top_network_indexes)
        self.networkEP_scores.append([numpy.mean(last_networks[x].score) for x in top_network_indexes])
        self.networkEP_completions.append([numpy.mean(last_networks[x].completions) for x in top_network_indexes])
    def expansion(self):
        self.a= len(self.networks[-1])
        for network in range(self.a):
            self.networks[-1].append(subCDE_Policy(self.state_size, self.action_size, self.parameters, self.evaluation_mode, freeze=False, initialweights=self.networks[-1][network].get_weigths()))
            self.networks[-1][-1].set_parameters(self.networks[-1][network].qnetwork_local,
                                                self.networks[-1][network].qnetwork_target,
                                                self.networks[-1][network].optimizer, self.networks[-1][network].ewc_loss, self.networks[-1][network].memory,
                                                self.networks[-1][network].loss, self.networks[-1][network].params,
                                                self.networks[-1][network].p_old)
            self.networks[-1][network].update_ewc()
            self.networks[-1][network].freeze = True
    def miniexpansion(self):
        for network in range(self.a):
            self.networks[-1].append(subCDE_Policy(self.state_size, self.action_size, self.parameters, self.evaluation_mode, freeze=False, initialweights=self.networks[-1][network].get_weigths()))
            self.networks[-1][-1].set_parameters(self.networks[-1][network].qnetwork_local,
                                                self.networks[-1][network].qnetwork_target,
                                                self.networks[-1][network].optimizer, self.networks[-1][network].ewc_loss, self.networks[-1][network].memory,
                                                self.networks[-1][network].loss, self.networks[-1][network].params,
                                                self.networks[-1][network].p_old)

    def set_evaluation_mode(self, evaluation_mode):
        self.evaluation_mode = evaluation_mode
    def get_name(self):
        return "CDE" + str(self.act_rotation)
    def get_activation(self):
        result = []
        for param_tuple in self.networks[-1][self.act_rotation].get_weigths():
            a, b = param_tuple
            result.append([a.tolist(), b.tolist()])
        return result
    def get_net(self):
        return self.act_rotation
    def get_fisher(self):
        return self.get_fisher_diag(self.qnetwork_local, self.memory, self.params)
class subCDE_Policy:
    def __init__(self, state_size, action_size, parameters, evaluation_mode=False, freeze=True, initialweights=0):
        self.evaluation_mode = evaluation_mode
        self.state_size = state_size
        self.action_size = action_size
        self.hidsize = parameters.hidden_size
        self.buffer_size = parameters.buffer_size
        self.batch_size = parameters.batch_size
        self.update_every = parameters.update_every
        self.learning_rate = parameters.learning_rate
        self.tau = parameters.tau
        self.gamma = parameters.gamma
        self.buffer_min_size = parameters.buffer_min_size
        self.freeze = freeze
        self.ewc_loss = 0
        self.ewc_lambda = parameters.ewc_lambda
        self.retain_graph = False
        self.score = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
        self.completions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.score_try = 0
        if initialweights == 0:
            a = torch.tensor((0.02996348, 0.61690165, 2.37539147, 3.06608078, 1.52474449, 0.25281987),dtype=torch.float), torch.tensor((1.19160814, 4.40811795, 0.91111034, 0.34885983),dtype=torch.float)
            self.weights = [a] * parameters.layer_count
        else:
            self.weights = copy.deepcopy(initialweights)

        if parameters.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("🐇 Using GPU")
        else:
            self.device = torch.device("cpu")
            print("🐢 Using CPU")

        self.qnetwork_local = DQN(state_size, action_size, self.weights, hidsize=self.hidsize).to(torch.device("cpu"))

        self.params = {n: p for n, p in self.qnetwork_local.named_parameters() if p.requires_grad}
        self.p_old = {}
        for n, p in copy.deepcopy(self.params).items():
            self.p_old[n] = torch.Tensor(p.data)

        self.qnetwork_local = self.qnetwork_local.to(self.device)
        self.qnetwork_target = copy.deepcopy(self.qnetwork_local).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, self.device)
        self.t_step = 0
    def set_parameters(self, ql, qt, opt, ewc_loss, buffer, loss, params, oldp):
        self.qnetwork_local.load_state_dict(ql.state_dict())
        self.qnetwork_target.load_state_dict(qt.state_dict())

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
        self.ewc_loss = ewc_loss
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.device)
        self.loss = loss
        self.params = params
        self.p_old = copy.deepcopy(oldp)
    def expansion(self):
        pass
    def act(self, handle, state, eps=0., eval = False):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state, self.freeze)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    def network_rotation(self, score, completions):
        pass
    def step(self, handle, state, action, reward, next_state, done):
        assert not self.evaluation_mode, "Policy has been initialized for evaluation only."
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.buffer_min_size and len(self.memory) > self.batch_size:
                self._learn()
    def _learn(self):
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences

        q_expected = self.qnetwork_local(states, self.freeze).gather(1, actions)
        q_targets_next = self.qnetwork_target(next_states, self.freeze).detach().max(1)[0].unsqueeze(-1)
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        q_expected = q_expected.to(self.device)
        q_targets = q_targets.to(self.device)

        if not self.ewc_loss == 0:
            self.ewc_loss = self.ewc_loss.to(self.device)

        self.loss = F.mse_loss(q_expected, q_targets) + self.ewc_lambda * self.ewc_loss

        self.loss = self.loss.to(self.device)

        self.optimizer.zero_grad()
        self.loss.backward(retain_graph=True)
        self.optimizer.step()
        self._soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
    def get_ewc_loss(self, model, fisher, p_old):
        loss = 0
        for n, p in model.named_parameters():
            if "activations" in n:
                continue
            _loss = fisher[n] * (p - p_old[n].to(self.device)) ** 2
            loss += _loss.sum()
        return loss
    def update_ewc(self):
        self.qnetwork_local = self.qnetwork_local.to(self.device)
        fisher_matrix = self.get_fisher_diag(self.qnetwork_local, self.memory, self.params)
        self.ewc_loss += self.get_ewc_loss(self.qnetwork_local, fisher_matrix, self.p_old).to(self.device)
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.device)

        last_device = self.device
        self.device = torch.device("cpu")
        self.retain_graph = True
        self.params = {n: p for n, p in self.qnetwork_local.named_parameters() if p.requires_grad}
        self.p_old = {n: p.clone().detach().to(self.device) for n, p in self.params.items()}

        self.qnetwork_local = self.qnetwork_local.to(last_device)
        self.device = last_device
    def get_fisher_diag(self, model, dataset, params):
        fisher = {}
        for n, p in copy.deepcopy(params).items():
            p.data.zero_()
            fisher[n] = Variable(p.data)

        model.eval()
        states, actions, rewards, next_states, done = dataset.sample()

        states = states.to(self.device)
        for x in range(states.size(dim=0)):
            model.zero_grad()
            output = model(states[x], self.freeze).view(1, -1)
            target = output.max(1)[1].view(-1)
            negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), target)
            negloglikelihood.backward(retain_graph=True)
        for n, p in model.named_parameters():
            if "activations" in n:
                continue
            fisher[n].data += p.grad.data ** 2 / len(dataset.sample())

        fisher = {n: p for n, p in fisher.items()}
        return fisher

    def _soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    def save_replay_buffer(self, filename):
        memory = self.memory.memory
        with open(filename, 'wb') as f:
            pickle.dump(list(memory)[-500000:], f)
    def load_replay_buffer(self, filename):
        with open(filename, 'rb') as f:
            self.memory.memory = pickle.load(f)
    def get_name(self):
        return "DQN"
    def get_weigths(self):
        return self.qnetwork_local.get_weights()
    def extract_nn_state_dict(self):
        return self.qnetwork_local.state_dict()
    def load_nn_state_dict(self, state_dict):
        self.qnetwork_local.load_state_dict(state_dict)
    def copy_params_from(self, other_dqn):
        self.load_state_dict(other_dqn.state_dict())
    def set_evaluation_mode(self, evaluation_mode):
        pass
    def reset_scores(self):
        pass
    def get_activation(self):
        result = []
        for param_tuple in self.get_weigths():
            a, b = param_tuple
            result.append([a.tolist(), b.tolist()])

        return result
    def get_net(self):
        return "1"

class DQN_Policy:
    def __init__(self, state_size, action_size, parameters, evaluation_mode=False, freeze=True, initialweights=0):
        self.evaluation_mode = evaluation_mode
        self.state_size = state_size
        self.action_size = action_size
        self.hidsize = parameters.hidden_size
        self.buffer_size = parameters.buffer_size
        self.batch_size = parameters.batch_size
        self.update_every = parameters.update_every
        self.learning_rate = parameters.learning_rate
        self.tau = parameters.tau
        self.gamma = parameters.gamma
        self.buffer_min_size = parameters.buffer_min_size
        self.freeze = freeze
        self.ewc_loss = 0
        self.ewc_lambda = 0.1
        self.retain_graph = False
        self.score = 0
        self.networkEP = []
        self.networkEP_scores = []
        self.networkEP_completions = []
        self.score_try = 0
        if initialweights == 0:
            a = torch.tensor((0.02996348, 0.61690165, 2.37539147, 3.06608078, 1.52474449, 0.25281987),dtype=torch.float), torch.tensor((1.19160814, 4.40811795, 0.91111034, 0.34885983),dtype=torch.float)
            self.weights = [a] * parameters.layer_count
        else:
            self.weights = copy.deepcopy(initialweights)

        if parameters.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("🐇 Using GPU")
        else:
            self.device = torch.device("cpu")
            print("🐢 Using CPU")

        self.qnetwork_local = DQN(state_size, action_size, self.weights, hidsize=self.hidsize).to(torch.device("cpu"))

        self.params = {n: p for n, p in self.qnetwork_local.named_parameters() if p.requires_grad}
        self.p_old = {}
        for n, p in copy.deepcopy(self.params).items():
            self.p_old[n] = torch.Tensor(p.data)

        self.qnetwork_local = self.qnetwork_local.to(self.device)
        self.qnetwork_target = copy.deepcopy(self.qnetwork_local).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, self.device)
        self.t_step = 0

    def expansion(self):
        pass
    def act(self, handle, state, eps=0., eval = False):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state, self.freeze)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    def network_rotation(self, score, completions):
        pass
    def step(self, handle, state, action, reward, next_state, done):
        assert not self.evaluation_mode, "Policy has been initialized for evaluation only."
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.buffer_min_size and len(self.memory) > self.batch_size:
                self._learn()
    def _learn(self):
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences

        q_expected = self.qnetwork_local(states, self.freeze).gather(1, actions)
        q_targets_next = self.qnetwork_target(next_states, self.freeze).detach().max(1)[0].unsqueeze(-1)
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        q_expected = q_expected.to(self.device)
        q_targets = q_targets.to(self.device)

        self.loss = F.mse_loss(q_expected, q_targets)
        self.loss = self.loss.to(self.device)

        self.optimizer.zero_grad()
        self.loss.backward(retain_graph=True)
        self.optimizer.step()
        self._soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
    def _soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    def get_name(self):
        return "DQN"
    def get_weigths(self):
        return self.qnetwork_local.get_weights()
    def set_evaluation_mode(self, evaluation_mode):
        pass
    def reset_scores(self):
        pass
    def get_activation(self):
        result = []
        for param_tuple in self.get_weigths():
            a, b = param_tuple
            result.append([a.tolist(), b.tolist()])

        return result
    def get_net(self):
        return "0"
class DQN_EWC_Policy:
    def __init__(self, state_size, action_size, parameters, evaluation_mode=False, freeze=True, initialweights=0):
        self.evaluation_mode = evaluation_mode
        self.state_size = state_size
        self.action_size = action_size
        self.hidsize = parameters.hidden_size
        self.buffer_size = parameters.buffer_size
        self.batch_size = parameters.batch_size
        self.update_every = parameters.update_every
        self.learning_rate = parameters.learning_rate
        self.tau = parameters.tau
        self.gamma = parameters.gamma
        self.buffer_min_size = parameters.buffer_min_size
        self.freeze = freeze
        self.ewc_loss = 0
        self.ewc_lambda = parameters.ewc_lambda
        self.retain_graph = False
        self.score = 0
        self.networkEP = []
        self.networkEP_scores = []
        self.networkEP_completions = []
        self.score_try = 0
        self.counter = 0
        if initialweights == 0:
            a = torch.tensor((0.02996348, 0.61690165, 2.37539147, 3.06608078, 1.52474449, 0.25281987),dtype=torch.float), torch.tensor((1.19160814, 4.40811795, 0.91111034, 0.34885983),dtype=torch.float)
            self.weights = [a] * parameters.layer_count
        else:
            self.weights = copy.deepcopy(initialweights)

        if parameters.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:2")
            print("🐇 Using GPU")
        else:
            self.device = torch.device("cpu")
            print("🐢 Using CPU")

        self.qnetwork_local = DQN(state_size, action_size, self.weights, hidsize=self.hidsize).to(torch.device("cpu"))

        self.params = {n: p for n, p in self.qnetwork_local.named_parameters() if p.requires_grad}
        self.p_old = {}
        for n, p in copy.deepcopy(self.params).items():
            self.p_old[n] = torch.Tensor(p.data)

        self.qnetwork_local = self.qnetwork_local.to(self.device)
        self.qnetwork_target = copy.deepcopy(self.qnetwork_local).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, self.device)
        self.t_step = 0
    def set_parameters(self, ql, qt, opt, ewc_loss, buffer, loss, params, oldp):
        self.qnetwork_local.load_state_dict(ql.state_dict())
        self.qnetwork_target.load_state_dict(qt.state_dict())

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
        self.ewc_loss = ewc_loss
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.device)
        self.loss = loss
        self.params = params
        self.p_old = copy.deepcopy(oldp)
    def expansion(self):
        self.update_ewc()
        self.fisher_info()
    def act(self, handle, state, eps=0., eval = False):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state, self.freeze)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    def network_rotation(self, score, completions):
        pass
    def step(self, handle, state, action, reward, next_state, done):
        assert not self.evaluation_mode, "Policy has been initialized for evaluation only."
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.buffer_min_size and len(self.memory) > self.batch_size:
                self._learn()
    def _learn(self):
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences

        q_expected = self.qnetwork_local(states, self.freeze).gather(1, actions)
        q_targets_next = self.qnetwork_target(next_states, self.freeze).detach().max(1)[0].unsqueeze(-1)
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        q_expected = q_expected.to(self.device)
        q_targets = q_targets.to(self.device)

        if not self.ewc_loss == 0:
            self.ewc_loss = self.ewc_loss.to(self.device)

        self.loss = F.mse_loss(q_expected, q_targets) + self.ewc_lambda * self.ewc_loss

        self.loss = self.loss.to(self.device)

        self.optimizer.zero_grad()
        self.loss.backward(retain_graph=True)
        self.optimizer.step()
        self._soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
    def get_ewc_loss(self, model, fisher, p_old):
        loss = 0
        for n, p in model.named_parameters():
            _loss = fisher[n] * (p - p_old[n].to(self.device)) ** 2
            loss += _loss.sum()
        return loss
    def update_ewc(self):
        self.qnetwork_local = self.qnetwork_local.to(self.device)
        self.fisher_matrix = self.get_fisher_diag(self.qnetwork_local, self.memory, self.params)
        self.ewc_loss += self.get_ewc_loss(self.qnetwork_local, self.fisher_matrix, self.p_old).to(self.device)
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.device)

        last_device = self.device
        self.device = torch.device("cpu")
        self.retain_graph = True
        self.params = {n: p for n, p in self.qnetwork_local.named_parameters() if p.requires_grad}
        self.p_old = {n: p.clone().detach().to(self.device) for n, p in self.params.items()}

        self.qnetwork_local = self.qnetwork_local.to(last_device)
        self.device = last_device
    def get_fisher_diag(self, model, dataset, params):
        fisher = {}
        for n, p in copy.deepcopy(params).items():
            p.data.zero_()
            fisher[n] = Variable(p.data)

        model.eval()
        states, actions, rewards, next_states, done = dataset.sample()

        states = states.to(self.device)
        for x in range(states.size(dim=0)):
            model.zero_grad()
            output = model(states[x], self.freeze).view(1, -1)
            target = output.max(1)[1].view(-1)
            negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), target)
            negloglikelihood.backward(retain_graph=True)
        for n, p in model.named_parameters():
            fisher[n].data += p.grad.data ** 2 / len(dataset.sample())

        fisher = {n: p for n, p in fisher.items()}
        return fisher
    def _soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    def save_replay_buffer(self, filename):
        memory = self.memory.memory
        with open(filename, 'wb') as f:
            pickle.dump(list(memory)[-500000:], f)
    def load_replay_buffer(self, filename):
        with open(filename, 'rb') as f:
            self.memory.memory = pickle.load(f)
    def get_name(self):
        return "DQNEWC"
    def get_weigths(self):
        return self.qnetwork_local.get_weights()
    def extract_nn_state_dict(self):
        """
        Extract the state dictionary of the Q-network from the current instance.
        """
        return self.qnetwork_local.state_dict()
    def load_nn_state_dict(self, state_dict):
        """
        Load the provided state dictionary into the Q-network of the current instance.
        """
        self.qnetwork_local.load_state_dict(state_dict)
    def copy_params_from(self, other_dqn):
        """
        Copy parameters from another DQN model.
        """
        self.load_state_dict(other_dqn.state_dict())
    def set_evaluation_mode(self, evaluation_mode):
        pass
    def reset_scores(self):
        pass
    def get_activation(self):
        result = []
        for param_tuple in self.get_weigths():
            a, b = param_tuple
            result.append([a.tolist(), b.tolist()])

        return result
    def get_net(self):
        return "0"
    def fisher_info(self):
        a =[]
        for x in self.fisher_matrix:
            if "layer" in x:
                b = []
                if "weight" in x:
                    for z in self.fisher_matrix[x]:
                        for t in z:
                            b.append(t.to('cpu').numpy())
                elif "bias" in x:
                    for z in self.fisher_matrix[x]:
                        b.append(z.to('cpu').numpy())
                a.append(b)
        pd.DataFrame(a).dropna().to_csv( str(self.counter) + '_fisher_info.csv', index=False, header=False)
        self.counter += 1
        return pd.DataFrame(a)
class DQN_PAU_Policy:
    def __init__(self, state_size, action_size, parameters, evaluation_mode=False, freeze=False, initialweights=0):
        self.evaluation_mode = evaluation_mode
        self.state_size = state_size
        self.action_size = action_size
        self.hidsize = parameters.hidden_size
        self.buffer_size = parameters.buffer_size
        self.batch_size = parameters.batch_size
        self.update_every = parameters.update_every
        self.learning_rate = parameters.learning_rate
        self.tau = parameters.tau
        self.gamma = parameters.gamma
        self.buffer_min_size = parameters.buffer_min_size
        self.freeze = freeze
        self.ewc_loss = 0
        self.ewc_lambda = 0.1
        self.retain_graph = False
        self.score = 0
        self.networkEP = []
        self.networkEP_scores = []
        self.networkEP_completions = []
        self.score_try = 0
        if initialweights == 0:
            a = torch.tensor((0.02996348, 0.61690165, 2.37539147, 3.06608078, 1.52474449, 0.25281987),dtype=torch.float), torch.tensor((1.19160814, 4.40811795, 0.91111034, 0.34885983),dtype=torch.float)
            self.weights = [a] * parameters.layer_count
        else:
            self.weights = copy.deepcopy(initialweights)

        if parameters.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:2")
            print("🐇 Using GPU")
        else:
            self.device = torch.device("cpu")
            print("🐢 Using CPU")

        self.qnetwork_local = DQN(state_size, action_size, self.weights, hidsize=self.hidsize).to(torch.device("cpu"))

        self.params = {n: p for n, p in self.qnetwork_local.named_parameters() if p.requires_grad}
        self.p_old = {}
        for n, p in copy.deepcopy(self.params).items():
            self.p_old[n] = torch.Tensor(p.data)

        self.qnetwork_local = self.qnetwork_local.to(self.device)
        self.qnetwork_target = copy.deepcopy(self.qnetwork_local).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, self.device)
        self.t_step = 0
    def set_parameters(self, ql, qt, opt, ewc_loss, buffer, loss, params, oldp):
        self.qnetwork_local.load_state_dict(ql.state_dict())
        self.qnetwork_target.load_state_dict(qt.state_dict())

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
        self.ewc_loss = ewc_loss
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.device)
        self.loss = loss
        self.params = params
        self.p_old = copy.deepcopy(oldp)
    def expansion(self):
        pass
    def act(self, handle, state, eps=0., eval = False):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state, self.freeze)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    def network_rotation(self, score, completions):
        pass
    def step(self, handle, state, action, reward, next_state, done):
        assert not self.evaluation_mode, "Policy has been initialized for evaluation only."
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.buffer_min_size and len(self.memory) > self.batch_size:
                self._learn()
    def _learn(self):
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences

        q_expected = self.qnetwork_local(states, self.freeze).gather(1, actions)
        q_targets_next = self.qnetwork_target(next_states, self.freeze).detach().max(1)[0].unsqueeze(-1)
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        q_expected = q_expected.to(self.device)
        q_targets = q_targets.to(self.device)

        if not self.ewc_loss == 0:
            self.ewc_loss = self.ewc_loss.to(self.device)

        self.loss = F.mse_loss(q_expected, q_targets) + self.ewc_lambda * self.ewc_loss

        self.loss = self.loss.to(self.device)

        self.optimizer.zero_grad()
        self.loss.backward(retain_graph=True)
        self.optimizer.step()
        self._soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
    def get_ewc_loss(self, model, fisher, p_old):
        loss = 0
        for n, p in model.named_parameters():
            _loss = fisher[n] * (p - p_old[n].to(self.device)) ** 2
            loss += _loss.sum()
        return loss
    def update_ewc(self):
        self.qnetwork_local = self.qnetwork_local.to(self.device)
        fisher_matrix = self.get_fisher_diag(self.qnetwork_local, self.memory, self.params)
        self.ewc_loss += self.get_ewc_loss(self.qnetwork_local, fisher_matrix, self.p_old).to(self.device)
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.device)

        last_device = self.device
        self.device = torch.device("cpu")
        self.retain_graph = True
        self.params = {n: p for n, p in self.qnetwork_local.named_parameters() if p.requires_grad}
        self.p_old = {n: p.clone().detach().to(self.device) for n, p in self.params.items()}

        self.qnetwork_local = self.qnetwork_local.to(last_device)
        self.device = last_device
    def get_fisher_diag(self, model, dataset, params):
        fisher = {}
        for n, p in copy.deepcopy(params).items():
            p.data.zero_()
            fisher[n] = Variable(p.data)

        model.eval()
        states, actions, rewards, next_states, done = dataset.sample()

        states = states.to(self.device)
        for x in range(states.size(dim=0)):
            model.zero_grad()
            output = model(states[x], self.freeze).view(1, -1)
            target = output.max(1)[1].view(-1)
            negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), target)
            negloglikelihood.backward(retain_graph=True)
        for n, p in model.named_parameters():
            fisher[n].data += p.grad.data ** 2 / len(dataset.sample())

        fisher = {n: p for n, p in fisher.items()}
        return fisher
    def _soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    def save_replay_buffer(self, filename):
        memory = self.memory.memory
        with open(filename, 'wb') as f:
            pickle.dump(list(memory)[-500000:], f)
    def load_replay_buffer(self, filename):
        with open(filename, 'rb') as f:
            self.memory.memory = pickle.load(f)
    def get_name(self):
        return "DQN"
    def get_weigths(self):
        return self.qnetwork_local.get_weights()
    def extract_nn_state_dict(self):
        """
        Extract the state dictionary of the Q-network from the current instance.
        """
        return self.qnetwork_local.state_dict()
    def load_nn_state_dict(self, state_dict):
        """
        Load the provided state dictionary into the Q-network of the current instance.
        """
        self.qnetwork_local.load_state_dict(state_dict)
    def copy_params_from(self, other_dqn):
        """
        Copy parameters from another DQN model.
        """
        self.load_state_dict(other_dqn.state_dict())
    def set_evaluation_mode(self, evaluation_mode):
        pass
    def reset_scores(self):
        pass
    def get_activation(self):
        result = []
        for param_tuple in self.get_weigths():
            a, b = param_tuple
            result.append([a.tolist(), b.tolist()])

        return result
    def get_net(self):
        return "0"
class DQN_EWC_PAU_Policy:
    def __init__(self, state_size, action_size, parameters, evaluation_mode=False, freeze=False, initialweights=0):
        self.evaluation_mode = evaluation_mode
        self.state_size = state_size
        self.action_size = action_size
        self.hidsize = parameters.hidden_size
        self.buffer_size = parameters.buffer_size
        self.batch_size = parameters.batch_size
        self.update_every = parameters.update_every
        self.learning_rate = parameters.learning_rate
        self.tau = parameters.tau
        self.gamma = parameters.gamma
        self.buffer_min_size = parameters.buffer_min_size
        self.freeze = freeze
        self.ewc_loss = 0
        self.ewc_lambda = parameters.ewc_lambda
        self.retain_graph = False
        self.score = 0
        self.networkEP = []
        self.networkEP_scores = []
        self.networkEP_completions = []
        self.score_try = 0
        self.counter = 0
        if initialweights == 0:
            a = torch.tensor((0.02996348, 0.61690165, 2.37539147, 3.06608078, 1.52474449, 0.25281987),dtype=torch.float), torch.tensor((1.19160814, 4.40811795, 0.91111034, 0.34885983),dtype=torch.float)
            self.weights = [a] * parameters.layer_count
        else:
            self.weights = copy.deepcopy(initialweights)

        if parameters.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("🐇 Using GPU")
        else:
            self.device = torch.device("cpu")
            print("🐢 Using CPU")

        self.qnetwork_local = DQN(state_size, action_size, self.weights, hidsize=self.hidsize).to(torch.device("cpu"))

        self.params = {n: p for n, p in self.qnetwork_local.named_parameters() if p.requires_grad}
        self.p_old = {}
        for n, p in copy.deepcopy(self.params).items():
            self.p_old[n] = torch.Tensor(p.data)

        self.qnetwork_local = self.qnetwork_local.to(self.device)
        self.qnetwork_target = copy.deepcopy(self.qnetwork_local).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, self.device)
        self.t_step = 0
    def set_parameters(self, ql, qt, opt, ewc_loss, buffer, loss, params, oldp):
        self.qnetwork_local.load_state_dict(ql.state_dict())
        self.qnetwork_target.load_state_dict(qt.state_dict())

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
        self.ewc_loss = ewc_loss
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.device)
        self.loss = loss
        self.params = params
        self.p_old = copy.deepcopy(oldp)
    def expansion(self):
        self.update_ewc()
        self.fisher_info()
    def act(self, handle, state, eps=0., eval = False):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state, self.freeze)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    def network_rotation(self, score, completions):
        pass
    def step(self, handle, state, action, reward, next_state, done):
        assert not self.evaluation_mode, "Policy has been initialized for evaluation only."
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.buffer_min_size and len(self.memory) > self.batch_size:
                self._learn()
    def _learn(self):
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences

        q_expected = self.qnetwork_local(states, self.freeze).gather(1, actions)
        q_targets_next = self.qnetwork_target(next_states, self.freeze).detach().max(1)[0].unsqueeze(-1)
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        q_expected = q_expected.to(self.device)
        q_targets = q_targets.to(self.device)

        if not self.ewc_loss == 0:
            self.ewc_loss = self.ewc_loss.to(self.device)

        self.loss = F.mse_loss(q_expected, q_targets) + self.ewc_lambda * self.ewc_loss

        self.loss = self.loss.to(self.device)

        self.optimizer.zero_grad()
        self.loss.backward(retain_graph=True)
        self.optimizer.step()
        self._soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
    def get_ewc_loss(self, model, fisher, p_old):
        loss = 0
        for n, p in model.named_parameters():
            _loss = fisher[n] * (p - p_old[n].to(self.device)) ** 2
            loss += _loss.sum()
        return loss
    def update_ewc(self):
        self.qnetwork_local = self.qnetwork_local.to(self.device)
        self.fisher_matrix = self.get_fisher_diag(self.qnetwork_local, self.memory, self.params)
        self.ewc_loss += self.get_ewc_loss(self.qnetwork_local, self.fisher_matrix, self.p_old).to(self.device)
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.device)

        last_device = self.device
        self.device = torch.device("cpu")
        self.retain_graph = True
        self.params = {n: p for n, p in self.qnetwork_local.named_parameters() if p.requires_grad}
        self.p_old = {n: p.clone().detach().to(self.device) for n, p in self.params.items()}

        self.qnetwork_local = self.qnetwork_local.to(last_device)
        self.device = last_device
    def get_fisher_diag(self, model, dataset, params):
        fisher = {}
        for n, p in copy.deepcopy(params).items():
            p.data.zero_()
            fisher[n] = Variable(p.data)

        model.eval()
        states, actions, rewards, next_states, done = dataset.sample()

        states = states.to(self.device)
        for x in range(states.size(dim=0)):
            model.zero_grad()
            output = model(states[x], self.freeze).view(1, -1)
            target = output.max(1)[1].view(-1)
            negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), target)
            negloglikelihood.backward(retain_graph=True)
        for n, p in model.named_parameters():
            fisher[n].data += p.grad.data ** 2 / len(dataset.sample())

        fisher = {n: p for n, p in fisher.items()}
        return fisher
    def _soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    def save_replay_buffer(self, filename):
        memory = self.memory.memory
        with open(filename, 'wb') as f:
            pickle.dump(list(memory)[-500000:], f)
    def load_replay_buffer(self, filename):
        with open(filename, 'rb') as f:
            self.memory.memory = pickle.load(f)
    def get_name(self):
        return "DQNEWC"
    def get_weigths(self):
        return self.qnetwork_local.get_weights()
    def extract_nn_state_dict(self):
        """
        Extract the state dictionary of the Q-network from the current instance.
        """
        return self.qnetwork_local.state_dict()
    def load_nn_state_dict(self, state_dict):
        """
        Load the provided state dictionary into the Q-network of the current instance.
        """
        self.qnetwork_local.load_state_dict(state_dict)
    def copy_params_from(self, other_dqn):
        """
        Copy parameters from another DQN model.
        """
        self.load_state_dict(other_dqn.state_dict())
    def set_evaluation_mode(self, evaluation_mode):
        pass
    def reset_scores(self):
        pass
    def get_activation(self):
        result = []
        for param_tuple in self.get_weigths():
            a, b = param_tuple
            result.append([a.tolist(), b.tolist()])

        return result
    def get_net(self):
        return "0"
    def fisher_info(self):
        a =[]
        for x in self.fisher_matrix:
            if "layer" in x:
                b = []
                if "weight" in x:
                    for z in self.fisher_matrix[x]:
                        for t in z:
                            b.append(t.to('cpu').numpy())
                elif "bias" in x:
                    for z in self.fisher_matrix[x]:
                        b.append(z.to('cpu').numpy())
                a.append(b)
        pd.DataFrame(a).dropna().to_csv( str(self.counter) + '_fisher_info.csv', index=False, header=False)
        self.counter += 1
        return pd.DataFrame(a)
class DQN_MAS_Policy:
    def __init__(self, state_size, action_size, parameters, evaluation_mode=False, freeze=True, initialweights=0):
        self.evaluation_mode = evaluation_mode
        self.state_size = state_size
        self.action_size = action_size
        self.hidsize = parameters.hidden_size
        self.buffer_size = parameters.buffer_size
        self.batch_size = parameters.batch_size
        self.update_every = parameters.update_every
        self.learning_rate = parameters.learning_rate
        self.tau = parameters.tau
        self.gamma = parameters.gamma
        self.buffer_min_size = parameters.buffer_min_size
        self.freeze = freeze
        self.ewc_loss = 0
        self.ewc_lambda = parameters.ewc_lambda
        self.retain_graph = False
        self.score = 0
        self.networkEP = []
        self.networkEP_scores = []
        self.networkEP_completions = []
        self.score_try = 0
        self.counter = 0
        self.retain_graph = False
        self.completions = 0

        self.score_try = 0
        if initialweights == 0:
            a = torch.tensor((0.02996348, 0.61690165, 2.37539147, 3.06608078, 1.52474449, 0.25281987),dtype=torch.float), torch.tensor((1.19160814, 4.40811795, 0.91111034, 0.34885983),dtype=torch.float)
            self.weights = [a] * parameters.layer_count
        else:
            self.weights = copy.deepcopy(initialweights)

        if parameters.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("🐇 Using GPU")
        else:
            self.device = torch.device("cpu")
            print("🐢 Using CPU")

        self.qnetwork_local = DQN(state_size, action_size, self.weights, hidsize=self.hidsize).to(torch.device("cpu"))

        self.params = {n: p for n, p in self.qnetwork_local.named_parameters() if p.requires_grad}
        self.p_old = {}
        for n, p in copy.deepcopy(self.params).items():
            self.p_old[n] = torch.Tensor(p.data)

        self.qnetwork_local = self.qnetwork_local.to(self.device)
        self.qnetwork_target = copy.deepcopy(self.qnetwork_local).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, self.device)
        self.t_step = 0
    def set_parameters(self, ql, qt, opt, ewc_loss, buffer, loss, params, oldp):
        self.qnetwork_local.load_state_dict(ql.state_dict())
        self.qnetwork_target.load_state_dict(qt.state_dict())

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
        self.ewc_loss = ewc_loss
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.device)
        self.loss = loss
        self.params = params
        self.p_old = copy.deepcopy(oldp)
    def expansion(self):
        self.update_mas()
    def act(self, handle, state, eps=0., eval = False):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state, self.freeze)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    def network_rotation(self, score, completions):
        pass
    def step(self, handle, state, action, reward, next_state, done):
        assert not self.evaluation_mode, "Policy has been initialized for evaluation only."
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.buffer_min_size and len(self.memory) > self.batch_size:
                self._learn()
    def _learn(self):
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences

        q_expected = self.qnetwork_local(states, self.freeze).gather(1, actions)
        q_targets_next = self.qnetwork_target(next_states, self.freeze).detach().max(1)[0].unsqueeze(-1)
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        q_expected = q_expected.to(self.device)
        q_targets = q_targets.to(self.device)

        if not self.ewc_loss == 0:
            self.ewc_loss = self.ewc_loss.to(self.device)

        self.loss = F.mse_loss(q_expected, q_targets) + self.ewc_lambda * self.ewc_loss

        self.loss = self.loss.to(self.device)

        self.optimizer.zero_grad()
        self.loss.backward(retain_graph=True)
        self.optimizer.step()
        self._soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
    def update_mas(self):
        """
        Updates the MAS loss by computing the importance weights (omega) and retaining previous parameter values.
        """
        self.qnetwork_local = self.qnetwork_local.to(self.device)
        omega = self.compute_importance_weights(self.qnetwork_local, self.memory, self.params)
        self.ewc_loss += self.compute_mas_loss(self.qnetwork_local, omega, self.p_old).to(self.device)
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.device)

        # Save current parameters as old parameters for future tasks
        last_device = self.device
        self.device = torch.device("cpu")
        self.retain_graph = True
        self.params = {n: p for n, p in self.qnetwork_local.named_parameters() if p.requires_grad}
        self.p_old = {n: p.clone().detach().to(self.device) for n, p in self.params.items()}

        self.qnetwork_local = self.qnetwork_local.to(last_device)
        self.device = last_device

    def compute_mas_loss(self, model, omega, p_old):
        """
        https://github.com/rahafaljundi/MAS-Memory-Aware-Synapses/blob/master/MAS_to_be_published/MAS.py
        Compute the MAS loss based on importance weights (omega) and previous parameter values.
        """
        loss = 0
        for n, p in model.named_parameters():
            if "activations" in n:
                continue
            _loss = omega[n] * (p - p_old[n].to(self.device)) ** 2
            loss += _loss.sum()
        return loss

    def compute_importance_weights(self, model, dataset, params):
        """
        Compute the importance weights (omega) for model parameters.
        """
        omega = {}
        for n, p in copy.deepcopy(params).items():
            p.data.zero_()
            omega[n] = Variable(p.data)

        model.eval()
        states, actions, rewards, next_states, done = dataset.sample()

        states = states.to(self.device)
        for x in range(states.size(dim=0)):
            model.zero_grad()
            output = model(states[x], self.freeze).view(1, -1)
            target = output.max(1)[1].view(-1)
            negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), target)
            negloglikelihood.backward(retain_graph=True)

        for n, p in model.named_parameters():
            if "activations" in n:
                continue
            omega[n].data += p.grad.data ** 2 / len(dataset.sample())

        omega = {n: p for n, p in omega.items()}
        return omega
    def _soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    def get_name(self):
        return "DQNMAS"
    def get_weigths(self):
        return self.qnetwork_local.get_weights()


    def get_activation(self):
        result = []
        for param_tuple in self.get_weigths():
            a, b = param_tuple
            result.append([a.tolist(), b.tolist()])

        return result
    def get_net(self):
        return "DQNMAS"
class DQN_SI_Policy:
    def __init__(self, state_size, action_size, parameters, evaluation_mode=False, freeze=True, initialweights=0):
        self.evaluation_mode = evaluation_mode
        self.state_size = state_size
        self.action_size = action_size
        self.hidsize = parameters.hidden_size
        self.buffer_size = parameters.buffer_size
        self.batch_size = parameters.batch_size
        self.update_every = parameters.update_every
        self.learning_rate = parameters.learning_rate
        self.tau = parameters.tau
        self.gamma = parameters.gamma
        self.buffer_min_size = parameters.buffer_min_size
        self.freeze = freeze
        self.si_lambda = parameters.ewc_lambda  # Regularization weight for SI
        self.si_omega = defaultdict(lambda: torch.zeros_like(torch.empty(0)))  # Importance weights
        self.si_params = {}
        self.si_prev_params = {}
        self.loss = 0
        self.retain_graph = False
        self.score = 0
        self.networkEP = []
        self.networkEP_scores = []
        self.networkEP_completions = []
        self.score_try = 0
        self.counter = 0
        # Initialize network weights
        if initialweights == 0:
            a = torch.tensor((0.02996348, 0.61690165, 2.37539147, 3.06608078, 1.52474449, 0.25281987), dtype=torch.float), \
                torch.tensor((1.19160814, 4.40811795, 0.91111034, 0.34885983), dtype=torch.float)
            self.weights = [a] * parameters.layer_count
        else:
            self.weights = copy.deepcopy(initialweights)

        # Device configuration
        if parameters.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("🐇 Using GPU")
        else:
            self.device = torch.device("cpu")
            print("🐢 Using CPU")

        # Initialize Q-networks
        self.qnetwork_local = DQN(state_size, action_size, self.weights, hidsize=self.hidsize).to(self.device)
        self.qnetwork_target = copy.deepcopy(self.qnetwork_local).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, self.device)
        self.t_step = 0

        # Save initial parameter state for SI
        self._save_initial_params()

    def _save_initial_params(self):
        """Save the initial parameters for SI tracking."""
        self.si_params = {name: p.clone().detach() for name, p in self.qnetwork_local.named_parameters() if p.requires_grad}
        self.si_prev_params = copy.deepcopy(self.si_params)
        self.si_omega = {name: torch.zeros_like(p) for name, p in self.si_params.items()}

    def update_si(self):
        """Update SI importance weights and save current parameter state."""
        for name, param in self.qnetwork_local.named_parameters():
            if param.requires_grad:
                # Accumulate importance weights based on parameter updates
                delta_param = param.data - self.si_prev_params[name]
                self.si_omega[name] += (delta_param ** 2).detach() / (1e-6 + delta_param.abs().sum())
                # Update the previous parameter state
                self.si_prev_params[name] = param.data.clone().detach()

    def compute_si_loss(self):
        """Compute the SI regularization loss."""
        si_loss = 0
        for name, param in self.qnetwork_local.named_parameters():
            if param.requires_grad:
                delta_param = param - self.si_params[name]
                si_loss += (self.si_omega[name] * delta_param ** 2).sum()
        return self.si_lambda * si_loss

    def expansion(self):
        """Handle task change in expansion."""
        # Finalize importance weights for the previous task
        self.update_si()
        # Save the current parameters as the baseline for the new task
        self._save_initial_params()
    def act(self, handle, state, eps=0.0, eval=False):
        """Select an action based on the current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state, self.freeze)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, handle, state, action, reward, next_state, done):
        """Save experience in replay memory and learn every `update_every` steps."""
        assert not self.evaluation_mode, "Policy has been initialized for evaluation only."
        self.memory.add(state, action, reward, next_state, done)

        # Learn every update_every steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.buffer_min_size and len(self.memory) > self.batch_size:
                self._learn()

    def _learn(self):
        """Sample a batch from memory and perform learning with SI regularization."""
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences

        # Compute expected Q values
        q_expected = self.qnetwork_local(states, self.freeze).gather(1, actions)
        q_targets_next = self.qnetwork_target(next_states, self.freeze).detach().max(1)[0].unsqueeze(-1)
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        q_expected = q_expected.to(self.device)
        q_targets = q_targets.to(self.device)

        # Compute standard loss
        self.loss = F.mse_loss(q_expected, q_targets)

        # Add SI regularization loss
        si_loss = self.compute_si_loss()
        print(si_loss)
        total_loss = self.loss + si_loss

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Soft update target network
        self._soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

        # Update SI importance weights
        self.update_si()

    def _soft_update(self, local_model, target_model, tau):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


    def get_name(self):
        return "DQNSI"
    def get_weigths(self):
        return self.qnetwork_local.get_weights()
    def network_rotation(self, score, completions):
        pass

    def get_activation(self):
        result = []
        for param_tuple in self.get_weigths():
            a, b = param_tuple
            result.append([a.tolist(), b.tolist()])

        return result
    def get_net(self):
        return "DQNSI"
class DQN_CBP_Policy:
    def __init__(self, state_size, action_size, parameters, evaluation_mode=False, freeze=True, initialweights=0):
        self.evaluation_mode = evaluation_mode
        self.state_size = state_size
        self.action_size = action_size
        self.hidsize = parameters.hidden_size
        self.buffer_size = parameters.buffer_size
        self.batch_size = parameters.batch_size
        self.update_every = parameters.update_every
        self.learning_rate = parameters.learning_rate
        self.tau = parameters.tau
        self.gamma = parameters.gamma
        self.buffer_min_size = parameters.buffer_min_size
        self.freeze = freeze
        self.si_lambda = parameters.ewc_lambda  # Regularization weight for SI
        self.si_omega = defaultdict(lambda: torch.zeros_like(torch.empty(0)))  # Importance weights
        self.si_params = {}
        self.si_prev_params = {}
        self.loss = 0
        self.retain_graph = False
        self.score = 0
        self.networkEP = []
        self.networkEP_scores = []
        self.networkEP_completions = []
        self.score_try = 0
        self.counter = 0
        if initialweights == 0:
            a = torch.tensor((0.02996348, 0.61690165, 2.37539147, 3.06608078, 1.52474449, 0.25281987),
                             dtype=torch.float), \
                torch.tensor((1.19160814, 4.40811795, 0.91111034, 0.34885983), dtype=torch.float)
            self.weights = [a] * parameters.layer_count
        else:
            self.weights = copy.deepcopy(initialweights)

        # Device configuration
        if parameters.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("🐇 Using GPU")
        else:
            self.device = torch.device("cpu")
            print("🐢 Using CPU")

        # Initialize Q-networks
        self.qnetwork_local = DQN(state_size, action_size, self.weights, hidsize=self.hidsize).to(self.device)
        self.qnetwork_target = copy.deepcopy(self.qnetwork_local).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, self.device)
        self.t_step = 0


    def expansion(self):
        pass
    def act(self, handle, state, eps=0.0, eval=False):
        """Select an action based on the current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state, self.freeze)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, handle, state, action, reward, next_state, done):
        """Save experience in replay memory and learn every `update_every` steps."""
        assert not self.evaluation_mode, "Policy has been initialized for evaluation only."
        self.memory.add(state, action, reward, next_state, done)

        # Learn every update_every steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.buffer_min_size and len(self.memory) > self.batch_size:
                self._learn()

    def _learn(self):
        """Sample a batch from memory and perform learning with SI regularization."""
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences

        # Compute expected Q values
        q_expected = self.qnetwork_local(states, self.freeze).gather(1, actions)
        q_targets_next = self.qnetwork_target(next_states, self.freeze).detach().max(1)[0].unsqueeze(-1)
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        q_expected = q_expected.to(self.device)
        q_targets = q_targets.to(self.device)

        # Compute standard loss
        self.loss = F.mse_loss(q_expected, q_targets)

        total_loss = self.loss

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Soft update target network
        self._soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        self._reset_neurons(reset_probability=1e-4)

    def _soft_update(self, local_model, target_model, tau):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def _reset_neurons(self, reset_probability=0.1):
        """
        Randomly reinitialize neurons in the local network with a certain probability.

        Args:
            reset_probability (float): Probability of resetting each neuron.
        """
        for layer in self.qnetwork_local.modules():
            if isinstance(layer, nn.Linear):  # Targeting linear (fully connected) layers
                num_neurons = layer.weight.size(0)

                # Iterate through each neuron (row) in the weight matrix
                for i in range(num_neurons):
                    # Determine whether to reset the neuron with the given probability
                    if torch.rand(1).item() < reset_probability:
                        print(f"Resetting neuron {i} in layer {layer}")
                        with torch.no_grad():
                            # Reinitialize the entire row of weights for the selected neuron
                            layer.weight[i] = nn.init.xavier_uniform_(torch.empty_like(layer.weight[i].unsqueeze(0)))
                            if layer.bias is not None:
                                layer.bias[i] = nn.init.zeros_(layer.bias[i].unsqueeze(0))



    def get_name(self):
        return "DQNCBP"

    def network_rotation(self, score, completions):
        pass

    def get_activation(self):
        return "Fill"
    def get_net(self):
        return "DQNCBP"


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
class DQN_test_Policy:
    def __init__(self, state_size, action_size, parameters, evaluation_mode=False, freeze=True, initialweights=0):
        self.evaluation_mode = evaluation_mode
        self.state_size = state_size
        self.action_size = action_size
        self.hidsize = parameters.hidden_size
        self.buffer_size = parameters.buffer_size
        self.batch_size = parameters.batch_size
        self.update_every = parameters.update_every
        self.learning_rate = parameters.learning_rate
        self.tau = parameters.tau
        self.gamma = parameters.gamma
        self.buffer_min_size = parameters.buffer_min_size
        self.freeze = freeze
        self.loss = 0
        self.retain_graph = False
        self.score = 0
        self.score_try = 0
        self.counter = 0
        # Initialize network weights
        if initialweights == 0:
            a = torch.tensor((0.02996348, 0.61690165, 2.37539147, 3.06608078, 1.52474449, 0.25281987), dtype=torch.float), \
                torch.tensor((1.19160814, 4.40811795, 0.91111034, 0.34885983), dtype=torch.float)
            self.weights = [a] * parameters.layer_count
        else:
            self.weights = copy.deepcopy(initialweights)

        # Device configuration
        if parameters.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("🐇 Using GPU")
        else:
            self.device = torch.device("cpu")
            print("🐢 Using CPU")

        # Initialize Q-networks
        self.qnetwork_local = DQN(state_size, action_size, self.weights, hidsize=self.hidsize).to(self.device)
        self.qnetwork_target = copy.deepcopy(self.qnetwork_local).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, self.device)
        self.t_step = 0
    def act(self, handle, state, eps=0.0, eval=False):
        """Select an action based on the current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state, self.freeze)
        self.qnetwork_local.train()
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, handle, state, action, reward, next_state, done):
        """Save experience in replay memory and learn every `update_every` steps."""
        assert not self.evaluation_mode, "Policy has been initialized for evaluation only."
        self.memory.add(state, action, reward, next_state, done)

        # Learn every update_every steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.buffer_min_size and len(self.memory) > self.batch_size:
                self._learn()

    def _learn(self):
        """Sample a batch from memory and perform learning with SI regularization."""
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences
        # Compute expected Q values
        q_expected = self.qnetwork_local(states, self.freeze).gather(1, actions)
        q_targets_next = self.qnetwork_target(next_states, self.freeze).detach().max(1)[0].unsqueeze(-1)
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        q_expected = q_expected.to(self.device)
        q_targets = q_targets.to(self.device)
        # Compute standard loss
        self.loss = F.mse_loss(q_expected, q_targets)
        total_loss = self.loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self._soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)


    def _soft_update(self, local_model, target_model, tau):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)