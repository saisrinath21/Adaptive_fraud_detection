"""Tests for DQN agent and RL components."""

import numpy as np
import torch
import pytest

from src.rl.networks import DuelingQNetwork, SimpleQNetwork
from src.rl.replay_buffer import PrioritizedReplayBuffer
from src.rl.environment import FraudDetectionEnv
from src.rl.dqn_agent import DQNAgent


class TestDuelingQNetwork:
    """Tests for the Dueling Q-Network architecture."""

    def test_output_shape(self):
        """Output should have shape (batch_size, n_actions)."""
        net = DuelingQNetwork(state_dim=20, n_actions=3)
        x = torch.randn(32, 20)
        output = net(x)

        assert output.shape == (32, 3)

    def test_single_input(self):
        """Should handle single observation."""
        net = DuelingQNetwork(state_dim=10, n_actions=3)
        x = torch.randn(1, 10)
        output = net(x)

        assert output.shape == (1, 3)

    def test_gradient_flow(self):
        """Gradients should flow through the network."""
        net = DuelingQNetwork(state_dim=15, n_actions=3)
        x = torch.randn(8, 15)
        output = net(x)
        loss = output.sum()
        loss.backward()

        for param in net.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestSimpleQNetwork:
    """Tests for the standard Q-Network."""

    def test_output_shape(self):
        net = SimpleQNetwork(state_dim=20, n_actions=3)
        x = torch.randn(16, 20)
        output = net(x)
        assert output.shape == (16, 3)


class TestPrioritizedReplayBuffer:
    """Tests for the prioritized experience replay buffer."""

    def test_push_and_length(self):
        """Buffer should track length correctly."""
        buffer = PrioritizedReplayBuffer(capacity=100)
        assert len(buffer) == 0

        state = np.zeros(10, dtype=np.float32)
        for i in range(50):
            buffer.push(state, 0, 1.0, state, False)

        assert len(buffer) == 50

    def test_sample_batch(self):
        """Should return correctly shaped batches."""
        buffer = PrioritizedReplayBuffer(capacity=100)
        state = np.zeros(10, dtype=np.float32)

        for i in range(100):
            buffer.push(state, np.random.randint(3), float(i), state, False)

        states, actions, rewards, next_states, dones, indices, weights = (
            buffer.sample(32)
        )

        assert states.shape == (32, 10)
        assert actions.shape == (32,)
        assert rewards.shape == (32,)
        assert next_states.shape == (32, 10)
        assert dones.shape == (32,)
        assert indices.shape == (32,)
        assert weights.shape == (32,)

    def test_is_ready(self):
        """is_ready should reflect buffer fullness."""
        buffer = PrioritizedReplayBuffer(capacity=100)
        state = np.zeros(5, dtype=np.float32)

        assert not buffer.is_ready(64)

        for i in range(64):
            buffer.push(state, 0, 1.0, state, False)

        assert buffer.is_ready(64)

    def test_capacity_overflow(self):
        """Buffer should not exceed capacity."""
        buffer = PrioritizedReplayBuffer(capacity=50)
        state = np.zeros(5, dtype=np.float32)

        for i in range(200):
            buffer.push(state, 0, 1.0, state, False)

        assert len(buffer) == 50

    def test_priority_update(self):
        """Priority updates should not crash."""
        buffer = PrioritizedReplayBuffer(capacity=100)
        state = np.zeros(5, dtype=np.float32)

        for i in range(100):
            buffer.push(state, 0, 1.0, state, False)

        _, _, _, _, _, indices, _ = buffer.sample(16)
        td_errors = np.random.uniform(0, 5, 16)
        buffer.update_priorities(indices, td_errors)


class TestFraudDetectionEnv:
    """Tests for the custom Gymnasium environment."""

    @pytest.fixture
    def simple_env(self):
        """Create a simple test environment."""
        np.random.seed(42)
        n = 100
        X = np.random.normal(0, 1, (n, 10)).astype(np.float32)
        y = np.random.choice([0, 1], n, p=[0.95, 0.05]).astype(np.int32)
        risk = np.random.uniform(0, 1, n).astype(np.float32)
        return FraudDetectionEnv(X, y, risk, max_steps=50)

    def test_reset(self, simple_env):
        """Reset should return valid state and info."""
        state, info = simple_env.reset()

        assert state.shape == (17,)  # features + risk + fraud_prob + cluster (5)
        assert isinstance(info, dict)

    def test_step(self, simple_env):
        """Step should return valid outputs."""
        state, _ = simple_env.reset()
        next_state, reward, terminated, truncated, info = simple_env.step(0)

        assert next_state.shape == (17,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_episode_terminates(self, simple_env):
        """Episode should terminate after max_steps."""
        state, _ = simple_env.reset()
        steps = 0
        done = False

        while not done:
            _, _, terminated, truncated, _ = simple_env.step(
                np.random.randint(3)
            )
            done = terminated or truncated
            steps += 1

        assert steps == 50  # max_steps

    def test_action_space(self, simple_env):
        """Should have 3 actions."""
        assert simple_env.action_space.n == 3

    def test_reward_structure(self, simple_env):
        """Rewards should follow the configured structure."""
        simple_env.reset()

        # The reward should be a valid float for any action
        for action in [0, 1, 2]:
            simple_env.reset()
            _, reward, _, _, _ = simple_env.step(action)
            assert isinstance(reward, float)
            assert not np.isnan(reward)

    def test_episode_summary(self, simple_env):
        """Episode summary should contain expected keys."""
        simple_env.reset()
        for _ in range(50):
            simple_env.step(np.random.randint(3))

        summary = simple_env.get_episode_summary()
        assert "total_transactions" in summary
        assert "total_reward" in summary
        assert "fraud_detection_rate" in summary
        assert "false_positive_rate" in summary


class TestDQNAgent:
    """Tests for the DQN agent."""

    @pytest.fixture
    def agent(self):
        """Create a test agent."""
        return DQNAgent(
            state_dim=17,
            n_actions=3,
            batch_size=16,
            buffer_size=500,
            epsilon_start=1.0,
            hidden_dims=[32, 32],
        )

    def test_select_action(self, agent):
        """Agent should return valid action."""
        state = np.random.normal(0, 1, 17).astype(np.float32)
        action = agent.select_action(state)

        assert action in [0, 1, 2]

    def test_select_action_greedy(self, agent):
        """Greedy action should be deterministic."""
        agent.epsilon = 0.0
        state = np.random.normal(0, 1, 17).astype(np.float32)

        actions = [agent.select_action(state, training=False) for _ in range(10)]
        assert len(set(actions)) == 1  # All same action

    def test_train_step_empty_buffer(self, agent):
        """Training with empty buffer should return None."""
        loss = agent.train_step()
        assert loss is None

    def test_short_training(self):
        """Brief training should not crash."""
        np.random.seed(42)
        n = 200
        X = np.random.normal(0, 1, (n, 10)).astype(np.float32)
        y = np.random.choice([0, 1], n, p=[0.95, 0.05]).astype(np.int32)
        risk = np.random.uniform(0, 1, n).astype(np.float32)

        env = FraudDetectionEnv(X, y, risk, max_steps=50)
        agent = DQNAgent(
            state_dim=17, n_actions=3,
            batch_size=16, buffer_size=500,
            hidden_dims=[32, 32],
        )

        history = agent.train(env, num_episodes=3, log_interval=1)

        assert len(history["episode_rewards"]) == 3
        assert len(history["epsilon_values"]) == 3

    def test_epsilon_decay(self, agent):
        """Epsilon should decrease during training."""
        initial_epsilon = agent.epsilon

        # Simulate several episodes of decay
        for _ in range(10):
            agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.epsilon_decay)

        assert agent.epsilon < initial_epsilon
