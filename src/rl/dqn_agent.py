"""
DQN Agent Module
=================
Double Dueling DQN agent with prioritized experience replay for
fraud detection decision-making. The agent learns an optimal policy
for approving, reviewing, or blocking transactions.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple
from collections import deque

from src.rl.networks import DuelingQNetwork
from src.rl.replay_buffer import PrioritizedReplayBuffer
from src.rl.environment import FraudDetectionEnv
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DQNAgent:
    """
    Double Dueling DQN Agent for fraud detection.

    Combines:
    - Dueling architecture (separate value and advantage streams)
    - Double DQN (decoupled action selection and evaluation)
    - Prioritized experience replay (focus on informative transitions)
    - Epsilon-greedy exploration with decay

    Parameters
    ----------
    state_dim : int
        Dimension of the state space.
    n_actions : int
        Number of possible actions (3: approve, review, block).
    learning_rate : float
        Adam optimizer learning rate.
    gamma : float
        Discount factor for future rewards.
    epsilon_start : float
        Initial exploration rate.
    epsilon_end : float
        Minimum exploration rate.
    epsilon_decay : float
        Multiplicative decay per episode.
    batch_size : int
        Training batch size.
    buffer_size : int
        Replay buffer capacity.
    target_update_freq : int
        Steps between target network hard updates.
    tau : float
        Soft update interpolation coefficient.
    hidden_dims : list of int
        Hidden layer sizes for Q-network.
    dropout : float
        Dropout rate.
    device : str
        'cuda' or 'cpu'.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int = 3,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_size: int = 50000,
        target_update_freq: int = 100,
        tau: float = 0.005,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        device: Optional[str] = None,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau

        # Device selection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"  DQN Agent using device: {self.device}")

        hidden_dims = hidden_dims or [128, 128, 64]

        # Online Q-network (updated every training step)
        self.q_network = DuelingQNetwork(
            state_dim=state_dim,
            n_actions=n_actions,
            hidden_dims=hidden_dims,
            dropout=dropout,
        ).to(self.device)

        # Target Q-network (updated periodically for stability)
        self.target_network = DuelingQNetwork(
            state_dim=state_dim,
            n_actions=n_actions,
            hidden_dims=hidden_dims,
            dropout=dropout,
        ).to(self.device)

        # Initialize target with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Prioritized replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_size)

        # Huber loss for stability (less sensitive to outliers than MSE)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

        # Running state normalization
        self._state_mean = np.zeros(state_dim, dtype=np.float32)
        self._state_var = np.ones(state_dim, dtype=np.float32)
        self._state_count = 0

        # Training tracking
        self.training_step = 0
        self.episode_count = 0
        self.training_history: Dict[str, List] = {
            "episode_rewards": [],
            "episode_avg_rewards": [],
            "episode_lengths": [],
            "epsilon_values": [],
            "losses": [],
            "fraud_detection_rates": [],
            "false_positive_rates": [],
            "action_distributions": [],
        }

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.

        Parameters
        ----------
        state : np.ndarray
            Current state observation.
        training : bool
            If True, use epsilon-greedy. If False, always exploit.

        Returns
        -------
        int
            Selected action (0=approve, 1=review, 2=block).
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        norm_state = self._normalize_state(state)
        state_tensor = torch.FloatTensor(norm_state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            self.q_network.eval()
            q_values = self.q_network(state_tensor)
            self.q_network.train()

        return q_values.argmax(dim=1).item()

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize a state using running mean/std."""
        return (state - self._state_mean) / (np.sqrt(self._state_var) + 1e-8)

    def _update_state_stats(self, states: np.ndarray) -> None:
        """Update running mean/variance with a batch of states."""
        batch_mean = states.mean(axis=0)
        batch_var = states.var(axis=0)
        batch_count = states.shape[0]

        total_count = self._state_count + batch_count
        delta = batch_mean - self._state_mean

        new_mean = self._state_mean + delta * batch_count / max(total_count, 1)
        m_a = self._state_var * self._state_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self._state_count * batch_count / max(total_count, 1)
        new_var = m2 / max(total_count, 1)

        self._state_mean = new_mean.astype(np.float32)
        self._state_var = new_var.astype(np.float32)
        self._state_count = total_count

    def train_step(self) -> Optional[float]:
        """
        Perform one training step using a batch from replay buffer.

        Returns
        -------
        float or None
            Training loss, or None if buffer not ready.
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return None

        # Sample batch with priorities
        (
            states, actions, rewards, next_states, dones,
            indices, weights,
        ) = self.replay_buffer.sample(self.batch_size)

        # Update running normalization stats
        self._update_state_stats(states)

        # Normalize states
        states = (states - self._state_mean) / (np.sqrt(self._state_var) + 1e-8)
        next_states = (next_states - self._state_mean) / (np.sqrt(self._state_var) + 1e-8)

        # Clip rewards to prevent exploding Q-values
        rewards = np.clip(rewards, -1.0, 1.0)

        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        weights_t = torch.FloatTensor(weights).to(self.device)

        # Current Q-values for taken actions
        current_q = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN: use online net for action selection, target net for evaluation
        with torch.no_grad():
            next_actions = self.q_network(next_states_t).argmax(dim=1)
            next_q = self.target_network(next_states_t).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)
            # Clamp target to prevent runaway Q-values
            target_q = target_q.clamp(-100.0, 100.0)

        # Compute weighted Huber loss (more stable than MSE)
        td_errors = current_q - target_q
        loss = (weights_t * self.loss_fn(current_q, target_q)).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Tight gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update priorities in replay buffer
        td_errors_np = td_errors.detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors_np)

        # Soft update target network
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self._soft_update_target()

        return loss.item()

    def train(
        self,
        env: FraudDetectionEnv,
        num_episodes: int = 500,
        log_interval: int = 10,
    ) -> Dict[str, List]:
        """
        Train the agent over multiple episodes.

        Parameters
        ----------
        env : FraudDetectionEnv
            The fraud detection environment.
        num_episodes : int
            Number of episodes to train.
        log_interval : int
            Episodes between logging progress.

        Returns
        -------
        dict
            Training history with metrics per episode.
        """
        logger.info("=" * 60)
        logger.info("DQN TRAINING")
        logger.info("=" * 60)
        logger.info(f"  Episodes: {num_episodes}")
        logger.info(f"  State dim: {self.state_dim}, Actions: {self.n_actions}")
        logger.info(f"  Batch size: {self.batch_size}, Buffer: {self.replay_buffer.capacity}")

        best_reward = float("-inf")
        recent_rewards = deque(maxlen=50)
        episode_losses = []

        for episode in range(num_episodes):
            state, info = env.reset()
            episode_reward = 0
            episode_steps = 0

            done = False
            while not done:
                # Select and take action
                action = self.select_action(state, training=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Store transition
                self.replay_buffer.push(state, action, reward, next_state, done)

                # Train
                loss = self.train_step()
                if loss is not None:
                    episode_losses.append(loss)

                state = next_state
                episode_reward += reward
                episode_steps += 1

            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            self.episode_count += 1

            # Get episode summary
            summary = env.get_episode_summary()

            # Track metrics
            self.training_history["episode_rewards"].append(episode_reward)
            self.training_history["episode_avg_rewards"].append(
                episode_reward / max(1, episode_steps)
            )
            self.training_history["episode_lengths"].append(episode_steps)
            self.training_history["epsilon_values"].append(self.epsilon)
            self.training_history["fraud_detection_rates"].append(
                summary["fraud_detection_rate"]
            )
            self.training_history["false_positive_rates"].append(
                summary["false_positive_rate"]
            )
            self.training_history["action_distributions"].append(
                dict(summary["action_counts"])
            )

            if episode_losses:
                avg_loss = np.mean(episode_losses[-100:])
                self.training_history["losses"].append(avg_loss)
            else:
                self.training_history["losses"].append(0.0)

            recent_rewards.append(episode_reward)

            # Track best model
            if episode_reward > best_reward:
                best_reward = episode_reward

            # Periodic logging
            if (episode + 1) % log_interval == 0:
                avg_recent = np.mean(recent_rewards)
                avg_loss = self.training_history["losses"][-1]
                fdr = summary["fraud_detection_rate"]
                fpr = summary["false_positive_rate"]

                logger.info(
                    f"  Episode {episode + 1}/{num_episodes} | "
                    f"Reward: {episode_reward:.1f} | "
                    f"Avg(50): {avg_recent:.1f} | "
                    f"ε: {self.epsilon:.3f} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"FDR: {fdr:.1%} | "
                    f"FPR: {fpr:.1%}"
                )

        logger.info(f"\n  Training complete. Best episode reward: {best_reward:.1f}")
        return self.training_history

    def evaluate(self, env: FraudDetectionEnv, n_episodes: int = 5) -> Dict:
        """
        Evaluate the agent without exploration.

        Parameters
        ----------
        env : FraudDetectionEnv
            Environment to evaluate in.
        n_episodes : int
            Number of evaluation episodes.

        Returns
        -------
        dict
            Evaluation metrics.
        """
        logger.info("Evaluating agent (greedy policy)...")

        all_rewards = []
        all_fdr = []
        all_fpr = []
        all_actions = []

        for ep in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = self.select_action(state, training=False)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                all_actions.append(action)

            summary = env.get_episode_summary()
            all_rewards.append(episode_reward)
            all_fdr.append(summary["fraud_detection_rate"])
            all_fpr.append(summary["false_positive_rate"])

        from collections import Counter
        action_dist = Counter(all_actions)
        total_actions = len(all_actions)

        results = {
            "mean_reward": np.mean(all_rewards),
            "std_reward": np.std(all_rewards),
            "mean_fraud_detection_rate": np.mean(all_fdr),
            "mean_false_positive_rate": np.mean(all_fpr),
            "action_distribution": {
                FraudDetectionEnv.ACTION_NAMES[a]: count / total_actions
                for a, count in action_dist.items()
            },
        }

        logger.info(f"  Mean reward: {results['mean_reward']:.1f} ± {results['std_reward']:.1f}")
        logger.info(f"  Fraud Detection Rate: {results['mean_fraud_detection_rate']:.1%}")
        logger.info(f"  False Positive Rate: {results['mean_false_positive_rate']:.1%}")
        logger.info(f"  Action distribution: {results['action_distribution']}")

        return results

    def _soft_update_target(self) -> None:
        """Soft-update target network: θ_target = τ·θ_online + (1-τ)·θ_target."""
        for target_param, online_param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1 - self.tau) * target_param.data
            )

    def set_epsilon(self, epsilon: float) -> None:
        """Manually set epsilon (used during drift adaptation)."""
        self.epsilon = epsilon
        logger.info(f"  Epsilon manually set to {epsilon:.3f}")

    def save(self, filepath: str) -> None:
        """Save agent state to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_step": self.training_step,
            "episode_count": self.episode_count,
            "training_history": self.training_history,
        }, filepath)
        logger.info(f"  Agent saved to: {filepath}")

    def load(self, filepath: str) -> None:
        """Load agent state from disk."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.training_step = checkpoint["training_step"]
        self.episode_count = checkpoint["episode_count"]
        self.training_history = checkpoint.get("training_history", self.training_history)
        logger.info(f"  Agent loaded from: {filepath}")
