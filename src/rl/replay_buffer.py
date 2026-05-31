"""
Prioritized Experience Replay Buffer
======================================
Stores (state, action, reward, next_state, done) transitions with
priority-based sampling. Higher priority given to rare fraud detections
and high-error transitions.
"""

import numpy as np
from typing import Tuple, Optional
from collections import namedtuple

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class SumTree:
    """
    Binary sum tree for efficient priority-based sampling.

    Each leaf stores a transition priority. Parent nodes store the
    sum of their children, enabling O(log n) sampling proportional
    to priorities.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write_ptr = 0

    def _propagate(self, idx: int, change: float) -> None:
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Retrieve leaf index for a given cumulative sum value."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Return total sum of all priorities."""
        return self.tree[0]

    def add(self, priority: float, data: Transition) -> None:
        """Add a transition with given priority."""
        idx = self.write_ptr + self.capacity - 1

        self.data[self.write_ptr] = data
        self.update(idx, priority)

        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float) -> None:
        """Update priority at a given tree index."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, Transition]:
        """Get (tree_index, priority, data) for a given cumulative sum."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer using a sum tree.

    Higher-priority transitions (e.g., rare fraud events, high TD-error)
    are sampled more frequently. Importance-sampling weights correct for
    the resulting bias.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    alpha : float
        Priority exponent (0 = uniform, 1 = full prioritization).
    beta_start : float
        Initial importance-sampling exponent (annealed to 1.0).
    beta_frames : int
        Number of frames over which to anneal beta.
    epsilon : float
        Small constant added to priorities to prevent zero-probability.
    """

    def __init__(
        self,
        capacity: int = 50000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        epsilon: float = 1e-6,
    ):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.frame = 0
        self.max_priority = 1.0

    @property
    def beta(self) -> float:
        """Current importance-sampling beta (annealed from beta_start to 1.0)."""
        return min(
            1.0,
            self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames,
        )

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store a transition with maximum priority (will be updated after training).
        """
        transition = Transition(state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, transition)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample a batch of transitions proportional to priorities.

        Returns
        -------
        tuple of:
            states, actions, rewards, next_states, dones,
            indices (for priority updates), weights (importance sampling)
        """
        batch_size = min(batch_size, self.tree.n_entries)
        self.frame += 1

        indices = np.zeros(batch_size, dtype=int)
        priorities = np.zeros(batch_size, dtype=np.float64)
        transitions = []

        # Divide the total priority range into equal segments
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s = np.random.uniform(lo, hi)
            idx, priority, data = self.tree.get(s)
            indices[i] = idx
            priorities[i] = priority
            transitions.append(data)

        # Importance-sampling weights
        probs = priorities / (self.tree.total() + 1e-10)
        weights = (self.tree.n_entries * probs) ** (-self.beta)
        weights = weights / (weights.max() + 1e-10)  # Normalize

        # Unpack transitions
        states = np.array([t.state for t in transitions], dtype=np.float32)
        actions = np.array([t.action for t in transitions], dtype=np.int64)
        rewards = np.array([t.reward for t in transitions], dtype=np.float32)
        next_states = np.array([t.next_state for t in transitions], dtype=np.float32)
        dones = np.array([t.done for t in transitions], dtype=np.float32)

        return states, actions, rewards, next_states, dones, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """
        Update priorities based on TD-errors from training.

        Parameters
        ----------
        indices : np.ndarray
            Tree indices of sampled transitions.
        td_errors : np.ndarray
            Absolute TD-errors from the last training step.
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return self.tree.n_entries

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for a batch."""
        return len(self) >= batch_size
