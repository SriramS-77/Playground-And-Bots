import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import os

# --- Constants ---
STATE_SIZE = 5  # [humanity, captchas_solved, req_rate_client, req_rate_server, total_users]
ACTION_SIZE = 11  # Threat levels 0-10
MEMORY_SIZE = 100000  # Size of the replay memory


class DQN(nn.Module):
    """Deep Q-Network for the expanded state."""

    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        # Define the network architecture
        self.layer1 = nn.Linear(state_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action_size)

    def forward(self, x):
        """Forward pass through the network."""
        # Ensure input is a float tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        elif x.dtype != torch.float32:
            x = x.float()

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class ThreatAssessor:
    """The RL Agent that learns to assess threats."""

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.95   #0.9995  # Slower decay for offline training
        self.learning_rate = 0.0005

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Offline RL Agent using device: {self.device}")

        # Create the policy and target networks
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only for evaluation

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        """Store an experience tuple in the replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state, use_exploration=True):
        """Select an action using an epsilon-greedy policy."""
        if use_exploration and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore: random action

        # Exploit: best action from the policy network
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return torch.argmax(q_values).item()  # .item() gets the integer value

    def train_model(self, batch_size):
        """Train the policy network using a batch of experiences from memory."""
        if len(self.memory) < batch_size:
            return  # Not enough experiences to train

        # Sample a random batch of experiences
        batch = random.sample(self.memory, batch_size)

        # Unzip the batch
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device).unsqueeze(1)

        # --- Q-Learning Update ---

        # 1. Get Q(s, a) for the current states and actions
        # We use .gather(1, actions) to select the Q-value for the specific action taken
        current_q_values = self.policy_net(states).gather(1, actions)

        # 2. Calculate the target Q-value: R + γ * max_a' Q_target(s', a')
        # We use the target_net for stability
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            # If it was a terminal state (done=1), the target is just the reward
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # 3. Calculate the loss (Mean Squared Error)
        loss = self.criterion(current_q_values, target_q_values)

        # 4. Perform backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon (exploration rate)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_net(self):
        """Update the target network to match the policy network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, filepath):
        """Save the agent's state (model weights, memory, etc.)."""
        print(f"Saving offline model state to {filepath}...")
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory': list(self.memory)  # Convert deque to list for saving
        }
        torch.save(checkpoint, filepath)
        print("Offline model saved.")

    def load_model(self, filepath):
        """Load the agent's state from a checkpoint."""
        if not os.path.exists(filepath):
            print(f"No offline model checkpoint found at {filepath}. Starting new model.")
            return

        try:
            print(f"Loading offline model from {filepath}...")
            checkpoint = torch.load(filepath, map_location=self.device)

            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.memory = deque(checkpoint['memory'], maxlen=MEMORY_SIZE)

            self.policy_net.to(self.device)
            self.target_net.to(self.device)
            self.target_net.eval()

            print("Offline model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}. Starting new model.")
            # Re-initialize a fresh agent
            self.memory = deque(maxlen=MEMORY_SIZE)
            self.policy_net = DQN(self.state_size, self.action_size).to(self.device)
            self.target_net = DQN(self.state_size, self.action_size).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)


# Create a single global instance of the agent
offline_rl_agent = ThreatAssessor(STATE_SIZE, ACTION_SIZE)
