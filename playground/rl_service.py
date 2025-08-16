import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np


class DQN(nn.Module):
    """
    A simple Deep Q-Network model.
    """

    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
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

    def train_model(self, batch_size=32):
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

            # Get the current Q-value prediction from the model
            current_q = self.model(state)[action]

            # Compute the target Q-value
            if done:
                target_q = reward
            else:
                next_q_values = self.model(next_state)
                target_q = reward + self.gamma * torch.max(next_q_values)

            # Calculate loss and perform backpropagation
            loss = self.criterion(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Decay epsilon to reduce exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        print(f"RL agent: Trained {len(minibatch)} batches")



# Instantiate a global agent
rl_agent = ThreatAssessor()
