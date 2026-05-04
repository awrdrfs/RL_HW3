import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from gridworld import GridWorld

# Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.model = QNetwork(state_dim, action_dim)
        self.target_model = QNetwork(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.buffer = ReplayBuffer(10000)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def update(self, batch_size):
        if len(self.buffer) < batch_size:
            return
        
        state, action, reward, next_state, done = self.buffer.sample(batch_size)
        
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)
        
        q_values = self.model(state).gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_state).max(1)[0]
        expected_q_values = reward + self.gamma * next_q_values * (1 - done)
        
        loss = nn.MSELoss()(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

def train():
    env = GridWorld(mode='static')
    agent = DQNAgent(16, 4)
    episodes = 501
    batch_size = 32
    target_update = 10
    
    rewards_history = []
    
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        while not done and step_count < 50:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.update(batch_size)
            step_count += 1
            
        if e % target_update == 0:
            agent.update_target_network()
            
        rewards_history.append(total_reward)
        if e % 50 == 0:
            print(f"Episode {e}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
            
    return agent, rewards_history

if __name__ == "__main__":
    train()
