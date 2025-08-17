import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os


class DQN(nn.Module):
    """
    A simple Deep Q-Network model.
    """

    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, output_size)
        )

    def forward(self, x):
        return self.network(x)


class ThreatAssessor:
    """
    Manages the RL agent's state, actions, and learning process.
    """

    def __init__(self):
        self.state_size = 2  # humanity_score, captchas_solved
        self.action_size = 11  # Threat levels 0-10
        self.memory = []
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.model = DQN(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        """Stores experience in the replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        """
        Selects an action (threat level) using an epsilon-greedy strategy.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.model(state_tensor)
        return np.argmax(action_values.cpu().data.numpy())

    def train_model(self, batch_size=16):
        """
        Trains the DQN model using a batch of experiences from memory.
        """
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            reward = torch.FloatTensor([reward])
            action = torch.LongTensor([action])

            current_q = self.model(state)[action]

            if done:
                target_q = reward
            else:
                next_q_values = self.model(next_state)
                target_q = reward + self.gamma * torch.max(next_q_values)

            loss = self.criterion(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.memory = self.memory[batch_size // 4:]

    def save_model(self, filepath="models/rl_model.pth"):
        """Saves the model and training state to a file."""
        print(f"Saving RL model state to {filepath}...")
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory': self.memory
        }
        torch.save(checkpoint, filepath)
        print("Model saved successfully.")

    def load_model(self, filepath="models/rl_model.pth"):
        """Loads the model and training state from a file."""
        if os.path.exists(filepath):
            print(f"Loading RL model state from {filepath}...")
            checkpoint = torch.load(filepath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.memory = checkpoint['memory']
            self.model.train()  # Set model to training mode
            print("Model loaded successfully.")
        else:
            print(f"No model found at {filepath}. Starting with a new model.")


# Instantiate a global agent
rl_agent = ThreatAssessor()
