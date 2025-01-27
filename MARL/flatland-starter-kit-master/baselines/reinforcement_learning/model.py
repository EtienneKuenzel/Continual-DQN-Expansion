import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union, Optional, Any
import torch.utils.checkpoint as checkpoint


class Network(nn.Module):

    def __init__(self, state_size, action_size, hidsize1=128, hidsize2=128):
        super(Network, self).__init__()
        self.fc1_adv = nn.Linear(state_size, hidsize1)
        self.fc2_adv = nn.Linear(hidsize1, hidsize2)
        self.fc3_adv = nn.Linear(hidsize2, action_size)

    def forward(self, x):
        x = F.relu(self.fc1_adv(x))
        x = F.relu(self.fc2_adv(x))
        return self.fc3_adv(x)

USE_CUDA = torch.cuda.is_available()

class DQN(nn.Module):
    #Dueling mit state in pau und adv normal
    def __init__(self,state_size, action_size,initial_weights, freeze_pau=False, hidsize=128):
        super(DQN, self).__init__()
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        # Create the first layer
        self.layers.append(nn.Linear(state_size, hidsize))
        nn.init.xavier_uniform_(self.layers[-1].weight, gain=nn.init.calculate_gain('linear'))

        # Create hidden layers
        for _ in range(len(initial_weights) - 1):
            self.layers.append(nn.Linear(hidsize, hidsize))
            nn.init.xavier_uniform_(self.layers[-1].weight, gain=nn.init.calculate_gain('linear'))

        # Create the output layer
        self.layers.append(nn.Linear(hidsize, action_size))
        nn.init.xavier_uniform_(self.layers[-1].weight, gain=nn.init.calculate_gain('linear'))

        # Create PAU activation functions
        for weights in initial_weights:
            self.activations.append(PAU(weights=weights, cuda=torch.cuda.is_available()).requires_grad_(not freeze_pau))



    def forward(self, state, freeze):
        if freeze:
            for x in self.activations:
                x.freeze()
        x = state
        for i in range(len(self.activations)):
            x = self.activations[i](self.layers[i](x))
        x = self.layers[-1](x)  # Output layer without activation
        return x
    def get_weights(self):
        weights_list = []
        for x in self.activations:
            a = x.weights_nominator, x.weights_denominator
            weights_list.append(a)
        return weights_list

__all__: List[str] = ["PAU", "freeze_pau"]


class PAU(nn.Module):
    """
    This class implements the Pade Activation Unit proposed in:
    https://arxiv.org/pdf/1907.06732.pdf
    """

    def __init__(
            self,
            weights,
            m: int = 5,
            n: int = 4,
            initial_shape: Optional[str] = "relu",
            efficient: bool = True,
            eps: float = 1e-08,
            activation_unit = 5 ,
            **kwargs: Any
    ) -> None:
        """
        Constructor method
        :param m (int): Size of nominator polynomial. Default 5.
        :param n (int): Size of denominator polynomial. Default 4.
        :param initial_shape (Optional[str]): Initial shape of PAU, if None random shape is used, also if m and n are
        not the default value (5 and 4) a random shape is utilized. Default "leaky_relu_0_2".
        :param efficient (bool): If true efficient variant with checkpointing is used. Default True.
        :param eps (float): Constant for numerical stability. Default 1e-08.
        :param **kwargs (Any): Unused
        """
        # Call super constructor
        super(PAU, self).__init__()
        # Save parameters
        self.efficient: bool = efficient
        self.m: int = m
        self.n: int = n
        self.eps: float = eps
        self.initial_weights = weights
        # Init weights
        weights_nominator, weights_denominator = self.initial_weights
        self.weights_nominator: Union[nn.Parameter, torch.Tensor] = nn.Parameter(weights_nominator.view(1, -1))
        self.weights_denominator: Union[nn.Parameter, torch.Tensor] = nn.Parameter(weights_denominator.view(1, -1))
    def freeze(self) -> None:
        """
        Function freezes the PAU weights by converting them to fixed model parameters.
        """

        if isinstance(self.weights_nominator, nn.Parameter):
            weights_nominator = self.weights_nominator.data.clone()
            del self.weights_nominator
            self.register_buffer("weights_nominator", weights_nominator)
        if isinstance(self.weights_denominator, nn.Parameter):
            weights_denominator = self.weights_denominator.data.clone()
            del self.weights_denominator
            self.register_buffer("weights_denominator", weights_denominator)
    def _forward(self,input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input (torch.Tensor): Input tensor of the shape [*]
        :return (torch.Tensor): Output tensor of the shape [*]
        """
        # Save original shape
        shape: Tuple[int, ...] = input.shape
        # Flatten input tensor
        input: torch.Tensor = input.view(-1)
        if self.efficient:
            # Init nominator and denominator
            nominator: torch.Tensor = torch.ones_like(input=input) * self.weights_nominator[..., 0]
            denominator: torch.Tensor = torch.zeros_like(input=input)
            # Compute nominator and denominator iteratively
            for index in range(1, self.m + 1):
                x: torch.Tensor = (input ** index)
                nominator: torch.Tensor = nominator + x * self.weights_nominator[..., index]
                if index < (self.n + 1):
                    denominator: torch.Tensor = denominator + x * self.weights_denominator[..., index - 1]
            denominator: torch.Tensor = denominator + 1.
        else:
            # Get Vandermonde matrix
            vander_matrix: torch.Tensor = torch.vander(x=input, N=self.m + 1, increasing=True)
            # Compute nominator
            nominator: torch.Tensor = (vander_matrix * self.weights_nominator).sum(-1)
            # Compute denominator
            denominator: torch.Tensor = 1. + torch.abs((vander_matrix[:, 1:self.n + 1]
                                                        * self.weights_denominator).sum(-1))
        # Compute output and reshape
        output: torch.Tensor = (nominator / denominator.clamp(min=self.eps)).view(shape)
        return output

    def forward(self,input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input (torch.Tensor): Input tensor of the shape [batch size, *]
        :return (torch.Tensor): Output tensor of the shape [batch size, *]
        """
        # Make input contiguous if needed
        input: torch.Tensor = input if input.is_contiguous() else input.contiguous()
        if self.efficient:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input=input)




def freeze_pau(module: Union[nn.Module, nn.DataParallel, nn.parallel.DataParallel]) -> Union[nn.Module, nn.DataParallel, nn.parallel.DataParallel]:
    """
    Function freezes any PAU in a given nn.Module (nn.DataParallel, nn.parallel.DataParallel).
    Function inspired by Timms function to freeze batch normalization layers.
    :param module (Union[nn.Module, nn.DataParallel, nn.parallel.DataParallel]): Original model with frozen PAU
    """
    res = module
    if isinstance(module, PAU):
        res.freeze()
    else:
        for name, child in module.named_children():
            print("g")
            new_child = freeze_pau(child)
            print(":(")
            if new_child is not child:
                res.add_module(name, new_child)
    return res

