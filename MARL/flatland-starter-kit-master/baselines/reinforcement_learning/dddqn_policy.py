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

from reinforcement_learning.model import DQN, Network
from reinforcement_learning.policy import Policy
torch.set_printoptions(precision=0)



class Continual_DQN_Expansion():
    """Continual DDQN Expansion(CDE)"""
    def __init__(self, state_size, action_size, parameters, evaluation_mode=False, freeze=True):
        self.networks = [[]]
        self.state_size = state_size
        self.action_size = action_size
        self.parameters = parameters
        self.evaluation_mode = evaluation_mode
        self.networks[0].append(DQNPolicy(state_size,action_size,parameters,evaluation_mode, initialweights=relu))
        self.x = 0
        self.act_rotation = 0
        self.evaluation_mode = False
        self.networkEP = [["N"],["E","P"]]


    def act(self, handle, state, eps=0.):
        if not self.evaluation_mode:
            return self.networks[-1][self.act_rotation].act(handle, state, eps)
        else:
            networks_copy = self.networks[-1][:]
            max_score = networks_copy[0].score
            max_score_index = 0
            a = -1
            for x in networks_copy:
                a += 1
                if max_score < x.score:
                    max_score_index = a
            self.networkEP[-1][max_score_index] = "+"
            return self.networks[-1][max_score_index].act(handle,state,eps)



    def network_rotation(self, score):
        self.networks[-1][self.act_rotation].score_try +=1
        self.networks[-1][self.act_rotation].score = (self.networks[-1][self.act_rotation].score + score)/ self.networks[-1][self.act_rotation].score_try
        self.act_rotation +=1
        self.act_rotation %= len(self.networks[-1])
    def step(self, handle, state, action, reward, next_state, done):
        #erstes netwrok wir im moment geupdate
        for network in self.networks[-1]:
            network.step(handle, state, action, reward, next_state, done)
    def expansion(self):
        self.act_rotation = 1
        #called with new Task
        networks_copy = self.networks[-1][:]
        max_score = networks_copy[0].score
        max_score_index = 0
        a = -1
        for x in networks_copy:
            a+=1
            if max_score < x.score:
                max_score_index = a
        if len(networks_copy) == 1:
            #adding the current network to the past stack
            self.networks.insert(len(self.networks) - 1, networks_copy[:])

            #aadding PAU to new stack
            self.networks[-1].append(DQNPolicy(self.state_size, self.action_size, self.parameters, self.evaluation_mode, freeze=False,initialweights=self.networks[-1][0].get_weigths()))
            self.networks[-1][1].set_parameters(self.networks[-1][0].qnetwork_local,
                                                self.networks[-1][0].qnetwork_target,
                                                self.networks[-1][0].optimizer, 0, self.networks[-1][0].memory,
                                                self.networks[-1][0].loss, self.networks[-1][0].params,
                                                self.networks[-1][0].p_old)

            #Adding EWC fertig to newstack
            self.networks[-1][0].update_ewc()
        else:
             if False:#max_score_index == 0:
                print("EWC ist besser")
                self.networkEP[-1][0] = self.networkEP[-1][0] + "+"
                self.networkEP.append(["EE","EP"])
                #add old networks to old stack
                self.networks.insert(len(self.networks) -1, networks_copy[:])
                # pau löschen
                self.networks[-1].pop()
                # Adding PAU Network fertig,
                self.networks[-1].append(DQNPolicy(self.state_size, self.action_size, self.parameters, self.evaluation_mode, freeze=False, initialweights=self.networks[-1][0].get_weigths()))
                self.networks[-1][1].set_parameters(self.networks[-1][0].qnetwork_local,
                                                    self.networks[-1][0].qnetwork_target,
                                                    self.networks[-1][0].optimizer, 0, self.networks[-1][0].memory,
                                                    self.networks[-1][0].loss, self.networks[-1][0].params,
                                                    self.networks[-1][0].p_old)

                # Adding EWC fertig
                self.networks[-1][0].update_ewc()
             else:
                print("PAU ist besser")
                self.networkEP[-1][1] = self.networkEP[-1][1] + "+"
                self.networkEP.append(["EE","EP","PE","PP","3P"])
                # add old networks to old stack
                self.networks.insert(len(self.networks) - 1, networks_copy[:])
                # pau löschen
                # Adding PAU Network fertig,
                #[[],[E,P],[E,P]]
                self.networks[-1].insert(1,DQNPolicy(self.state_size, self.action_size, self.parameters, self.evaluation_mode, freeze=False,initialweights=self.networks[-1][0].get_weigths()))
                self.networks[-1][1].set_parameters(self.networks[-1][0].qnetwork_local,
                                                    self.networks[-1][0].qnetwork_target,
                                                    self.networks[-1][0].optimizer, 0, self.networks[-1][0].memory,
                                                    self.networks[-1][0].loss, self.networks[-1][0].params,
                                                    self.networks[-1][0].p_old)
                #[[],[E,P],[E,EP, P]]
                # Adding EWC fertig
                self.networks[-1][0].update_ewc()
                #[[],[E,P],[EE,EP, P]]
                self.networks[-1].insert(3, DQNPolicy(self.state_size, self.action_size, self.parameters,self.evaluation_mode, freeze=False,initialweights=self.networks[-1][2].get_weigths()))
                self.networks[-1][3].set_parameters(self.networks[-1][2].qnetwork_local,
                                                    self.networks[-1][2].qnetwork_target,
                                                    self.networks[-1][2].optimizer, 0, self.networks[-1][2].memory,
                                                    self.networks[-1][2].loss, self.networks[-1][2].params,
                                                    self.networks[-1][2].p_old)
                #[[],[E,P],[EE,EP, PE, PP]]
                # Adding EWC fertig
                self.networks[-1][2].update_ewc()
                self.networks[-1][2].freeze = True
                #[[],[E,P],[EE,EP,PE,PP]]

                #Adding Theta3 pau gemischt mit ewc
                #claculating avg of weights and network params
                af_weight_theta1 = self.networks[-2][0].get_weigths()
                af_weight_theta2 = self.networks[-1][0].get_weigths()
                theta1_net_param = self.networks[-2][0].extract_nn_state_dict()
                theta2_net_param = self.networks[-1][0].extract_nn_state_dict()

                theta1_layer1, theta1_layer2 = af_weight_theta1
                theta2_layer1, theta2_layer2 = af_weight_theta2

                theta1_layer1_nom, theta1_layer1_denom = theta1_layer1
                theta1_layer2_nom, theta1_layer2_denom = theta1_layer2
                theta2_layer1_nom, theta2_layer1_denom = theta2_layer1
                theta2_layer2_nom, theta2_layer2_denom = theta2_layer2

                theta3_layer1 = (theta1_layer1_nom + theta2_layer1_nom)/2, (theta1_layer1_denom + theta2_layer1_denom)/2
                theta3_layer2 = (theta1_layer2_nom + theta2_layer2_nom)/2, (theta1_layer2_denom + theta2_layer2_denom)/2
                theta3_weigths = [theta3_layer1, theta3_layer2]
                theta3_weigths = [(t1.detach(), t2.detach()) for t1, t2 in theta3_weigths]
                self.networks[-1].append(DQNPolicy(self.state_size, self.action_size, self.parameters, self.evaluation_mode, freeze=False,initialweights=theta3_weigths))
                #adding avg network params
                averaged_state_dict = {}
                for key in theta1_net_param.keys():
                    averaged_state_dict[key] = (theta1_net_param[key] + theta2_net_param[key]) / 2
                self.networks[-1][-1].load_nn_state_dict(averaged_state_dict)
                # [[],[E,P],[EE,EP,PE,PP,3P]]
        self.reset_scores()
        print(self.networks[-1][0].score)


    def set_evaluation_mode(self, evaluation_mode):
        self.evaluation_mode = evaluation_mode
    def reset_scores(self):
        for x in self.networks[-1]:
            x.score = 0
            x.score_try = 0
    def get_name(self):
        return "CDE" + str(self.act_rotation)
    def get_expansion_code(self):
        return self.networkEP
a = torch.tensor((0.02996348, 0.61690165, 2.37539147, 3.06608078, 1.52474449, 0.25281987),dtype=torch.float), torch.tensor((1.19160814, 4.40811795, 0.91111034, 0.34885983),dtype=torch.float)
b = torch.tensor((0.02996348, 0.61690165, 2.37539147, 3.06608078, 1.52474449, 0.25281987),dtype=torch.float), torch.tensor((1.19160814, 4.40811795, 0.91111034, 0.34885983),dtype=torch.float)
relu = [a, b]


class DQNPolicy:
    def __init__(self, state_size, action_size, parameters, evaluation_mode=False, freeze=True, initialweights=relu):
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
        self.score_try = 0
        self.weights = initialweights

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
        self.qnetwork_local = ql
        self.qnetwork_target = qt
        self.optimizer = opt
        self.ewc_loss = ewc_loss
        self.memory = buffer
        self.loss = loss
        self.params = params
        self.p_old = oldp

    def expansion(self):
        self.update_ewc()

    def act(self, handle, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state, self.freeze)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def network_rotation(self, score):
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

    def save(self, filename):
        torch.save(self.qnetwork_local.state_dict(), filename + ".local")
        torch.save(self.qnetwork_target.state_dict(), filename + ".target")
    def load(self, filename):
        if os.path.exists(filename + ".local"):
            self.qnetwork_local.load_state_dict(torch.load(filename + ".local"))
        if os.path.exists(filename + ".target"):
            self.qnetwork_target.load_state_dict(torch.load(filename + ".target"))
    def save_replay_buffer(self, filename):
        memory = self.memory.memory
        with open(filename, 'wb') as f:
            pickle.dump(list(memory)[-500000:], f)
    def load_replay_buffer(self, filename):
        with open(filename, 'rb') as f:
            self.memory.memory = pickle.load(f)
    def test(self):
        self.act(np.array([[0] * self.state_size]))
        self._learn()
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
    def get_expansion_code(self):
        return [["DQNwithEWC"]]
class DDDQNPolicy_rpau(Policy):
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

        # Device
        if parameters.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:2")
            print("🐇 Using GPU")
        else:
            self.device = torch.device("cpu")
            print("🐢 Using CPU")

        # Q-Network
        print("a")
        self.qnetwork_local =Network(state_size, action_size, hidsize1=self.hidsize, hidsize2=self.hidsize).to(self.device)
        print("b")
        if not evaluation_mode:
            self.qnetwork_target = copy.deepcopy(self.qnetwork_local)
            self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
            self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, self.device)

            self.t_step = 0
            self.loss = 0.0

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
    def update_ewc(self):
        pass
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

        # Compute loss
        self.loss = F.mse_loss(q_expected, q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        # Update target network
        self._soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def _soft_update(self, local_model, target_model, tau):
        # Soft update model parameters.
        # θ_target = τ*θ_local + (1 - τ)*θ_target
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, filename):
        torch.save(self.qnetwork_local.state_dict(), filename + ".local")
        torch.save(self.qnetwork_target.state_dict(), filename + ".target")

    def load(self, filename):
        if os.path.exists(filename + ".local"):
            self.qnetwork_local.load_state_dict(torch.load(filename + ".local"))
        if os.path.exists(filename + ".target"):
            self.qnetwork_target.load_state_dict(torch.load(filename + ".target"))

    def save_replay_buffer(self, filename):
        memory = self.memory.memory
        with open(filename, 'wb') as f:
            pickle.dump(list(memory)[-500000:], f)

    def load_replay_buffer(self, filename):
        with open(filename, 'rb') as f:
            self.memory.memory = pickle.load(f)

    def test(self):
        self.act(np.array([[0] * self.state_size]))
        self._learn()
    def get_name(self):
        return "DDDQN + RPAU"



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
