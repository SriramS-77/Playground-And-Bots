import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import os

# --- Constants ---
STATE_SIZE = 5  # [humanity, avg_humanity, captchas_solved, last_solved_status, server_total_users]
ACTION_SIZE = 11  # Threat levels 0-10
MEMORY_SIZE = 10000 # Size per buffer (Human/Bot)

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(state_size, 128) # Increased width for better feature extraction
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, action_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)

class ThreatAssessor:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # --- BALANCED BUFFERS ---
        self.human_memory = deque(maxlen=MEMORY_SIZE)
        self.bot_memory = deque(maxlen=MEMORY_SIZE)

        self.gamma = 0.95 
        self.epsilon = 1.0 
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.0005

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.HuberLoss() # Huber is more robust to outliers than MSE

    def normalize_state(self, state):
        """Standardizes input features to [0, 1] or small ranges for NN stability."""
        state_copy = np.array(state, dtype=np.float32).copy()
        # [0]: humanity_score (already 0-1)
        # [1]: avg_humanity (already 0-1)
        # [2]: captchas_solved (0-10+) -> Squash with tanh
        state_copy[2] = np.tanh(state_copy[2] / 5.0)
        # [3]: last_solved_status (0 or 1)
        # [4]: total_users (0-300+) -> Normalize by 300
        state_copy[4] = state_copy[4] / 300.0
        return state_copy

    def remember(self, state, action, reward, next_state, done, is_bot):
        """Store experience in the appropriate buffer."""
        # Pre-normalize before storing to save compute during training
        s = self.normalize_state(state)
        ns = self.normalize_state(next_state)
        
        entry = (s, action, reward, ns, done)
        if is_bot:
            self.bot_memory.append(entry)
        else:
            self.human_memory.append(entry)

    def select_action(self, state, use_exploration=True):
        if use_exploration and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            s_norm = self.normalize_state(state)
            state_tensor = torch.tensor(s_norm, dtype=torch.float32).to(self.device).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return torch.argmax(q_values).item()

    def train_model(self, batch_size):
        """Ensures 50/50 Human/Bot split in every training batch."""
        half_batch = batch_size // 2
        
        if len(self.human_memory) < half_batch or len(self.bot_memory) < half_batch:
            return 

        # Sample 50% from each
        h_batch = random.sample(self.human_memory, half_batch)
        b_batch = random.sample(self.bot_memory, half_batch)
        combined_batch = h_batch + b_batch
        random.shuffle(combined_batch) # Shuffle so the network doesn't see all humans then all bots

        states, actions, rewards, next_states, dones = zip(*combined_batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device).unsqueeze(1)

        # DQN Logic
        current_q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_net(self):
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
            'human_memory': list(self.human_memory),  # Convert deque to list for saving
            'bot_memory': list(self.bot_memory)
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
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.human_memory = deque(checkpoint['human_memory'], maxlen=MEMORY_SIZE)
            self.bot_memory = deque(checkpoint['bot_memory'], maxlen=MEMORY_SIZE)

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
