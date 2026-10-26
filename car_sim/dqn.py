"""Double DQN in PyTorch: network, replay buffer, training step."""
import random

import numpy as np
import torch
from torch import nn

from car_sim import config


class QNetwork(nn.Module):
    """State (8 numbers) -> one Q value per action."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.STATE_DIM, config.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_SIZE, config.ACTION_DIM),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Fixed-size ring buffer; the oldest experiences are overwritten when full."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.states = np.zeros((capacity, config.STATE_DIM), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, config.STATE_DIM), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self._next = 0
        self._size = 0

    def push(self, state, action, reward, next_state, done):
        i = self._next
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self._next = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, n):
        idx = np.random.randint(0, self._size, size=n)
        return (
            torch.from_numpy(self.states[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.rewards[idx]),
            torch.from_numpy(self.next_states[idx]),
            torch.from_numpy(self.dones[idx]),
        )

    def __len__(self):
        return self._size


def compute_targets(online, target, rewards, next_states, dones):
    """Double DQN: the online net picks the next action, the target net scores it."""
    with torch.no_grad():
        best_actions = online(next_states).argmax(dim=1, keepdim=True)
        next_q = target(next_states).gather(1, best_actions).squeeze(1)
        return rewards + config.GAMMA * next_q * (1.0 - dones)


class DoubleDQNAgent:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.online = QNetwork().to(self.device)
        self.target = QNetwork().to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.LR)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(config.MEMORY_SIZE)

    def act(self, state, epsilon):
        """Epsilon-greedy: random action with probability epsilon, else the best Q value."""
        if random.random() < epsilon:
            return random.randrange(config.ACTION_DIM)
        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.online(x).argmax(dim=1).item())

    def act_batch(self, states, epsilon):
        """act() for many cars: one forward pass, then each car explores on its own."""
        with torch.no_grad():
            x = torch.as_tensor(np.stack(states), dtype=torch.float32, device=self.device)
            best = self.online(x).argmax(dim=1).tolist()
        return [random.randrange(config.ACTION_DIM) if random.random() < epsilon else int(a)
                for a in best]

    def remember(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        """One gradient step on a random batch. Returns the loss, or None if too little data."""
        if len(self.buffer) < config.BATCH_SIZE:
            return None
        states, actions, rewards, next_states, dones = (
            t.to(self.device) for t in self.buffer.sample(config.BATCH_SIZE)
        )
        q = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        targets = compute_targets(self.online, self.target, rewards, next_states, dones)
        loss = self.loss_fn(q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def update_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path, meta):
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "meta": dict(meta),
            },
            path,
        )

    def load(self, path):
        """Restore weights from a file and return the saved meta dict."""
        try:
            data = torch.load(path, map_location=self.device, weights_only=True)
            self.online.load_state_dict(data["online"])
            self.target.load_state_dict(data["target"])
            self.optimizer.load_state_dict(data["optimizer"])
            return dict(data["meta"])
        except Exception as exc:  # corrupt, truncated, or from a different network shape
            raise ValueError(f"Could not load model from {path}: {exc}") from exc
