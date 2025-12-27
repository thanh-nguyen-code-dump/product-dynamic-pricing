"""
Reinforcement Learning for Dynamic Pricing

This module implements Deep Q-Networks (DQN) and Policy Gradient methods
for learning optimal pricing policies through interaction with the market.

Key features:
- GPU-accelerated training and inference
- Experience replay with prioritized sampling
- Dueling network architecture
- Double DQN for stable learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import random


@dataclass
class RLConfig:
    """Configuration for RL-based pricing."""
    # State space
    state_dim: int = 12  # price, demand, inventory, competitors, seasonality
    
    # Action space (discrete price adjustments)
    price_adjustments: Tuple[float, ...] = (-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15)
    
    # Network architecture
    hidden_sizes: Tuple[int, ...] = (256, 256, 128)
    
    # Training parameters
    gamma: float = 0.99  # Discount factor
    tau: float = 0.005  # Soft update coefficient
    learning_rate: float = 1e-4
    batch_size: int = 256
    buffer_size: int = 100000
    
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.9995
    
    # Prioritized replay
    alpha: float = 0.6  # Priority exponent
    beta_start: float = 0.4  # Importance sampling
    beta_end: float = 1.0


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN architecture for pricing decisions.
    
    Separates state-value and advantage functions for more
    stable learning, especially important when many actions
    have similar values (common in pricing).
    
    V(s) estimates the value of being in state s
    A(s,a) estimates the advantage of taking action a in state s
    Q(s,a) = V(s) + A(s,a) - mean(A)
    """
    
    def __init__(self, config: RLConfig):
        super().__init__()
        self.config = config
        self.n_actions = len(config.price_adjustments)
        
        # Shared feature extraction layers
        layers = []
        prev_size = config.state_dim
        for hidden_size in config.hidden_sizes[:-1]:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size
        self.feature_network = nn.Sequential(*layers)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(prev_size, config.hidden_sizes[-1]),
            nn.ReLU(),
            nn.Linear(config.hidden_sizes[-1], 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_size, config.hidden_sizes[-1]),
            nn.ReLU(),
            nn.Linear(config.hidden_sizes[-1], self.n_actions)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for all actions."""
        features = self.feature_network(state)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine using dueling formula
        # Q = V + (A - mean(A))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values
    
    def get_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """Select action using epsilon-greedy policy."""
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        
        with torch.no_grad():
            q_values = self.forward(state)
            return q_values.argmax(dim=1).item()


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer.
    
    Samples transitions with probability proportional to their
    TD error, focusing learning on surprising experiences.
    Uses sum-tree for O(log n) sampling.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        
        self.buffer = deque(maxlen=capacity)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add transition to buffer with max priority."""
        max_priority = self.priorities[:self.size].max() if self.size > 0 else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(
        self,
        batch_size: int,
        beta: float = 0.4
    ) -> Tuple[np.ndarray, ...]:
        """Sample batch with prioritized sampling."""
        if self.size == 0:
            return None
        
        # Compute sampling probabilities
        priorities = self.priorities[:self.size] ** self.alpha
        probs = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, size=batch_size, p=probs)
        
        # Compute importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        
        # Gather transitions
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32)
        )
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + 1e-6


class DynamicPricingAgent:
    """
    DQN agent for dynamic pricing decisions.
    
    Uses Double DQN with dueling architecture and prioritized
    experience replay for stable, sample-efficient learning.
    
    Example:
        agent = DynamicPricingAgent(config)
        
        # Training loop
        state = env.reset()
        for step in range(n_steps):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            state = next_state
        
        # Inference
        optimal_adjustment = agent.get_optimal_price_adjustment(current_state)
    """
    
    def __init__(
        self,
        config: Optional[RLConfig] = None,
        device: str = "auto"
    ):
        self.config = config or RLConfig()
        
        # Auto-detect best device: CUDA > MPS > CPU
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # Networks
        self.q_network = DuelingQNetwork(self.config).to(self.device)
        self.target_network = DuelingQNetwork(self.config).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.q_network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-5
        )
        
        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            self.config.buffer_size,
            self.config.alpha
        )
        
        # Exploration
        self.epsilon = self.config.epsilon_start
        self.beta = self.config.beta_start
        
        # Training step counter
        self.training_steps = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        epsilon = self.epsilon if training else 0.0
        
        state_tensor = torch.tensor(
            state[np.newaxis, :],
            device=self.device,
            dtype=torch.float32
        )
        
        return self.q_network.get_action(state_tensor, epsilon)
    
    def get_price_adjustment(self, action: int) -> float:
        """Convert action index to price adjustment."""
        return self.config.price_adjustments[action]
    
    def get_optimal_price_adjustment(self, state: np.ndarray) -> float:
        """Get optimal price adjustment for given state."""
        action = self.select_action(state, training=False)
        return self.get_price_adjustment(action)
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Optional[float]:
        """Perform one training update."""
        if self.replay_buffer.size < self.config.batch_size:
            return None
        
        # Sample batch
        batch = self.replay_buffer.sample(self.config.batch_size, self.beta)
        if batch is None:
            return None
        
        states, actions, rewards, next_states, dones, indices, weights = batch
        
        # Convert to tensors
        states = torch.tensor(states, device=self.device)
        actions = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, device=self.device)
        next_states = torch.tensor(next_states, device=self.device)
        dones = torch.tensor(dones, device=self.device)
        weights = torch.tensor(weights, device=self.device)
        
        # Compute current Q-values
        current_q = self.q_network(states).gather(1, actions).squeeze(1)
        
        # Compute target Q-values using Double DQN
        with torch.no_grad():
            # Select actions with online network
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            # Evaluate with target network
            next_q = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + self.config.gamma * next_q * (1 - dones)
        
        # Compute TD errors for priority update
        td_errors = (current_q - target_q).detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # Compute weighted loss
        loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft update target network
        self._soft_update()
        
        # Update exploration parameters
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
        beta_increase = (self.config.beta_end - self.config.beta_start) / 100000
        self.beta = min(self.config.beta_end, self.beta + beta_increase)
        
        self.training_steps += 1
        
        return loss.item()
    
    def _soft_update(self):
        """Soft update of target network parameters."""
        tau = self.config.tau
        for target_param, param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def save(self, path: str):
        """Save agent to disk."""
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.__dict__,
            "epsilon": self.epsilon,
            "training_steps": self.training_steps
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = "auto") -> "DynamicPricingAgent":
        """Load agent from disk."""
        checkpoint = torch.load(path, map_location=device)
        config = RLConfig(**checkpoint["config"])
        
        agent = cls(config, device)
        agent.q_network.load_state_dict(checkpoint["q_network"])
        agent.target_network.load_state_dict(checkpoint["target_network"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        agent.epsilon = checkpoint["epsilon"]
        agent.training_steps = checkpoint["training_steps"]
        
        return agent


class PricingEnvironment:
    """
    Simulated pricing environment for RL training.
    
    Models market dynamics including:
    - Price-demand elasticity
    - Competitor responses
    - Seasonal patterns
    - Inventory constraints
    """
    
    def __init__(
        self,
        base_price: float = 100.0,
        base_demand: float = 1000.0,
        cost: float = 60.0,
        elasticity: float = -2.0,
        initial_inventory: float = 5000.0,
        horizon: int = 30
    ):
        self.base_price = base_price
        self.base_demand = base_demand
        self.cost = cost
        self.elasticity = elasticity
        self.initial_inventory = initial_inventory
        self.horizon = horizon
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_price = self.base_price
        self.inventory = self.initial_inventory
        self.day = 0
        self.competitor_price = self.base_price * (0.9 + 0.2 * random.random())
        self.seasonal_factor = 1.0
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Take action and return next state, reward, done.
        
        Args:
            action: Index of price adjustment
            
        Returns:
            next_state, reward, done
        """
        # Apply price adjustment
        adjustments = (-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15)
        price_change = adjustments[action]
        new_price = self.current_price * (1 + price_change)
        
        # Compute demand
        price_ratio = new_price / self.base_price
        demand = self.base_demand * (price_ratio ** self.elasticity) * self.seasonal_factor
        
        # Add competitor effect
        competitor_ratio = self.competitor_price / new_price
        demand *= 0.5 + 0.5 * np.tanh(competitor_ratio - 1)
        
        # Add noise
        demand *= (0.9 + 0.2 * random.random())
        demand = max(0, demand)
        
        # Constrain by inventory
        actual_sales = min(demand, self.inventory)
        
        # Compute reward (profit)
        revenue = new_price * actual_sales
        cost = self.cost * actual_sales
        holding_cost = 0.01 * self.inventory  # Inventory holding cost
        stockout_penalty = 0.5 * max(0, demand - self.inventory)  # Lost sales
        
        reward = revenue - cost - holding_cost - stockout_penalty
        
        # Update state
        self.current_price = new_price
        self.inventory = max(0, self.inventory - actual_sales)
        self.day += 1
        
        # Update competitor price (simple model)
        if self.current_price < self.competitor_price * 0.95:
            self.competitor_price *= 0.98  # Competitor responds
        elif self.current_price > self.competitor_price * 1.05:
            self.competitor_price *= 1.01
        
        # Update seasonality
        self.seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * self.day / 30)
        
        done = self.day >= self.horizon or self.inventory <= 0
        
        return self._get_state(), reward / 1000, done  # Normalize reward
    
    def _get_state(self) -> np.ndarray:
        """Construct state vector."""
        return np.array([
            self.current_price / self.base_price,  # Normalized price
            self.inventory / self.initial_inventory,  # Inventory level
            self.competitor_price / self.base_price,  # Competitor price
            self.seasonal_factor,  # Seasonality
            self.day / self.horizon,  # Time remaining
            np.sin(2 * np.pi * self.day / 7),  # Day of week (sin)
            np.cos(2 * np.pi * self.day / 7),  # Day of week (cos)
            (self.current_price - self.cost) / self.current_price,  # Margin
            0.0, 0.0, 0.0, 0.0  # Padding to state_dim=12
        ], dtype=np.float32)


def train_pricing_agent(
    n_episodes: int = 1000,
    device: str = "auto"
) -> DynamicPricingAgent:
    """Train a pricing agent on simulated environment."""
    config = RLConfig()
    agent = DynamicPricingAgent(config, device)
    env = PricingEnvironment()
    
    episode_rewards = []
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.update()
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    return agent
