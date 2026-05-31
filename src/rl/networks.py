"""
DQN Neural Network Architectures
==================================
Implements Dueling DQN with separate value and advantage streams
for more stable Q-value estimation in fraud detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DuelingQNetwork(nn.Module):
    """
    Dueling Deep Q-Network.

    Separates the Q-value into a state value V(s) and per-action
    advantage A(s, a), which improves learning stability especially
    when many actions have similar values.

    Q(s, a) = V(s) + A(s, a) - mean(A(s, ·))

    Parameters
    ----------
    state_dim : int
        Dimension of the input state.
    n_actions : int
        Number of possible actions.
    hidden_dims : list of int
        Sizes of hidden layers in the shared feature network.
    dropout : float
        Dropout rate for regularization.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int = 3,
        hidden_dims: List[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()

        hidden_dims = hidden_dims or [128, 128, 64]
        self.state_dim = state_dim
        self.n_actions = n_actions

        # Shared feature extraction layers
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims[:-1]:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        self.feature_net = nn.Sequential(*layers)

        # Value stream: estimates V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1),
        )

        # Advantage stream: estimates A(s, a) for each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], n_actions),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing Q-values for all actions.

        Parameters
        ----------
        state : torch.Tensor of shape (batch_size, state_dim)
            Input state.

        Returns
        -------
        torch.Tensor of shape (batch_size, n_actions)
            Q-values for each action.
        """
        features = self.feature_net(state)

        value = self.value_stream(features)              # (batch, 1)
        advantage = self.advantage_stream(features)      # (batch, n_actions)

        # Combine: Q = V + (A - mean(A))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Initialize network weights using He initialization."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


class SimpleQNetwork(nn.Module):
    """
    Standard (non-dueling) Q-Network for comparison/ablation.

    Parameters
    ----------
    state_dim : int
        Dimension of the input state.
    n_actions : int
        Number of possible actions.
    hidden_dims : list of int
        Sizes of hidden layers.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int = 3,
        hidden_dims: List[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()

        hidden_dims = hidden_dims or [128, 128, 64]

        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, n_actions))
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
