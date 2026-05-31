"""
Fraud Detection Gymnasium Environment
=======================================
Custom Gymnasium environment that presents transactions to the RL agent
one at a time. The agent observes transaction features + risk signals and
chooses to Approve, Review, or Block.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Optional, Tuple, Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FraudDetectionEnv(gym.Env):
    """
    Custom Gymnasium environment for fraud detection.

    State: Transaction features + clustering/anomaly signals
    Actions: 0=Approve, 1=Review, 2=Block
    Reward: Based on correctness of decision vs actual fraud label

    Parameters
    ----------
    X : np.ndarray of shape (n_transactions, n_features)
        Feature matrix including risk signals.
    y : np.ndarray of shape (n_transactions,)
        True fraud labels (0=legitimate, 1=fraud).
    risk_scores : np.ndarray of shape (n_transactions,)
        Precomputed risk scores for each transaction.
    fraud_probs : np.ndarray, optional
        Supervised fraud probabilities (LightGBM).
    cluster_features : np.ndarray, optional
        Per-transaction cluster feature matrix (n_samples, n_cluster_features).
    rewards_config : dict
        Reward values for different outcomes.
    max_steps : int
        Maximum number of transactions per episode.
    shuffle : bool
        Whether to shuffle transactions at each episode reset.
    """

    metadata = {"render_modes": ["human"]}

    # Action constants
    ACTION_APPROVE = 0
    ACTION_REVIEW = 1
    ACTION_BLOCK = 2
    ACTION_NAMES = {0: "APPROVE", 1: "REVIEW", 2: "BLOCK"}

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        risk_scores: np.ndarray,
        fraud_probs: Optional[np.ndarray] = None,
        cluster_features: Optional[np.ndarray] = None,
        rewards_config: Optional[Dict[str, float]] = None,
        max_steps: int = 1000,
        shuffle: bool = True,
    ):
        super().__init__()

        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int32)
        self.risk_scores = np.asarray(risk_scores, dtype=np.float32)
        self.fraud_probs = (
            np.asarray(fraud_probs, dtype=np.float32)
            if fraud_probs is not None
            else np.zeros(len(X), dtype=np.float32)
        )
        self.cluster_features = (
            np.asarray(cluster_features, dtype=np.float32)
            if cluster_features is not None
            else np.zeros((len(X), 5), dtype=np.float32)
        )
        self.max_steps = min(max_steps, len(X))
        self.shuffle = shuffle

        # State: features + risk + fraud_prob + cluster block
        self.state_dim = (
            self.X.shape[1] + 1 + 1 + self.cluster_features.shape[1]
        )
        self.n_actions = 3

        # Action and observation spaces
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

        # Reward configuration
        self.rewards = rewards_config or {
            "correct_fraud_block": 10.0,
            "correct_approve": 3.0,
            "correct_review_fraud": 5.0,
            "false_positive_block": -15.0,
            "missed_fraud": -25.0,
            "review_cost": -1.0,
        }

        # Episode tracking
        self.current_step = 0
        self.indices = np.arange(len(self.X))
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_outcomes = []

        # Metrics accumulators
        self.total_correct = 0
        self.total_missed_fraud = 0
        self.total_false_positive = 0

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment for a new episode."""
        super().reset(seed=seed)

        self.current_step = 0
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_outcomes = []
        self.total_correct = 0
        self.total_missed_fraud = 0
        self.total_false_positive = 0

        if self.shuffle:
            self.indices = self.np_random.permutation(len(self.X))
        else:
            self.indices = np.arange(len(self.X))

        state = self._get_state()
        info = self._get_info()

        return state, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Process one transaction decision.

        Parameters
        ----------
        action : int
            0=Approve, 1=Review, 2=Block

        Returns
        -------
        state : np.ndarray
            Next transaction state.
        reward : float
            Reward for the decision.
        terminated : bool
            Whether episode is complete.
        truncated : bool
            Whether episode was cut short.
        info : dict
            Additional information.
        """
        idx = self.indices[self.current_step]
        is_fraud = self.y[idx]

        # Compute reward based on action and true label
        reward = self._compute_reward(action, is_fraud)

        # Track outcomes
        self.episode_rewards.append(reward)
        self.episode_actions.append(action)
        outcome = self._classify_outcome(action, is_fraud)
        self.episode_outcomes.append(outcome)

        # Update metrics
        self._update_metrics(action, is_fraud)

        # Advance to next transaction
        self.current_step += 1

        # Check termination
        terminated = self.current_step >= self.max_steps
        truncated = self.current_step >= len(self.X)

        # Get next state
        if not (terminated or truncated):
            state = self._get_state()
        else:
            state = np.zeros(self.state_dim, dtype=np.float32)

        info = self._get_info()

        return state, reward, terminated, truncated, info

    def _get_state(self) -> np.ndarray:
        """Get current transaction state."""
        idx = self.indices[self.current_step]
        features = self.X[idx]
        risk = self.risk_scores[idx]
        fraud_p = self.fraud_probs[idx]
        cluster = self.cluster_features[idx]

        state = np.concatenate([features, [risk], [fraud_p], cluster]).astype(np.float32)

        # Handle any NaN/inf
        state = np.nan_to_num(state, nan=0.0, posinf=10.0, neginf=-10.0)

        return state

    def _compute_reward(self, action: int, is_fraud: int) -> float:
        """Compute reward for a given action and true label."""
        if is_fraud == 1:
            # Transaction is actually fraud
            if action == self.ACTION_BLOCK:
                return self.rewards["correct_fraud_block"]       # +10
            elif action == self.ACTION_REVIEW:
                return self.rewards["correct_review_fraud"]      # +5
            else:  # APPROVE
                return self.rewards["missed_fraud"]              # -25
        else:
            # Transaction is legitimate
            if action == self.ACTION_APPROVE:
                return self.rewards["correct_approve"]           # +3
            elif action == self.ACTION_REVIEW:
                return self.rewards["review_cost"]               # -1
            else:  # BLOCK
                return self.rewards["false_positive_block"]      # -15

    def _classify_outcome(self, action: int, is_fraud: int) -> str:
        """Classify the outcome of a decision."""
        if is_fraud:
            if action == self.ACTION_BLOCK:
                return "TRUE_POSITIVE"
            elif action == self.ACTION_REVIEW:
                return "REVIEW_FRAUD"
            else:
                return "MISSED_FRAUD"
        else:
            if action == self.ACTION_APPROVE:
                return "TRUE_NEGATIVE"
            elif action == self.ACTION_REVIEW:
                return "REVIEW_LEGIT"
            else:
                return "FALSE_POSITIVE"

    def _update_metrics(self, action: int, is_fraud: int) -> None:
        """Update running metrics."""
        if is_fraud and action == self.ACTION_BLOCK:
            self.total_correct += 1
        elif not is_fraud and action == self.ACTION_APPROVE:
            self.total_correct += 1
        elif is_fraud and action == self.ACTION_APPROVE:
            self.total_missed_fraud += 1
        elif not is_fraud and action == self.ACTION_BLOCK:
            self.total_false_positive += 1

    def _get_info(self) -> dict:
        """Get current episode information."""
        n_steps = max(1, len(self.episode_rewards))
        return {
            "step": self.current_step,
            "total_reward": sum(self.episode_rewards),
            "avg_reward": sum(self.episode_rewards) / n_steps,
            "action_distribution": {
                name: self.episode_actions.count(a) / n_steps
                for a, name in self.ACTION_NAMES.items()
            } if self.episode_actions else {},
            "accuracy": self.total_correct / n_steps,
            "missed_fraud": self.total_missed_fraud,
            "false_positives": self.total_false_positive,
        }

    def get_episode_summary(self) -> dict:
        """Get comprehensive summary of the completed episode."""
        from collections import Counter
        outcome_counts = Counter(self.episode_outcomes)
        n = len(self.episode_outcomes)

        return {
            "total_transactions": n,
            "total_reward": sum(self.episode_rewards),
            "avg_reward": sum(self.episode_rewards) / max(1, n),
            "outcomes": dict(outcome_counts),
            "action_counts": Counter(self.episode_actions),
            "fraud_detection_rate": (
                outcome_counts.get("TRUE_POSITIVE", 0) /
                max(1, outcome_counts.get("TRUE_POSITIVE", 0) + outcome_counts.get("MISSED_FRAUD", 0))
            ),
            "false_positive_rate": (
                outcome_counts.get("FALSE_POSITIVE", 0) /
                max(1, outcome_counts.get("TRUE_NEGATIVE", 0) + outcome_counts.get("FALSE_POSITIVE", 0))
            ),
        }
