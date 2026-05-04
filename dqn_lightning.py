import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
import random
from collections import deque
from gridworld import GridWorld
from dqn_variants import DuelingQNetwork, ReplayBuffer

class DQNLiteModule(pl.LightningModule):
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        super(DQNLiteModule, self).__init__()
        self.save_hyperparameters()
        
        self.model = DuelingQNetwork(state_dim, action_dim)
        self.target_model = DuelingQNetwork(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.buffer = ReplayBuffer(20000)
        self.epsilon = epsilon_start
        self.env = GridWorld(mode='random')
        self.state = self.env.reset()
        self.total_reward = 0
        self.episode_reward = 0
        self.current_step = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.hparams.action_dim - 1)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def training_step(self, batch, batch_idx):
        # 採樣一步
        action = self.select_action(self.state)
        next_state, reward, done = self.env.step(action)
        self.current_step += 1
        
        # 如果超過 50 步也強制結束
        if self.current_step >= 50:
            done = True
            
        self.buffer.push(self.state, action, reward, next_state, done)
        self.state = next_state
        self.episode_reward += reward
        
        if done:
            self.log('episode_reward', self.episode_reward, prog_bar=True)
            self.episode_reward = 0
            self.state = self.env.reset()
            self.current_step = 0
            
        if len(self.buffer) < 128:
            return None # 還沒開始訓練
            
        # 從 Buffer 採樣 Batch
        states, actions, rewards, next_states, dones = self.buffer.sample(128)
        
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            # 使用 Double DQN 邏輯
            next_actions = self.model(next_states).argmax(1).unsqueeze(1)
            next_q_values = self.target_model(next_states).gather(1, next_actions).squeeze(1)
            expected_q_values = rewards + self.hparams.gamma * next_q_values * (1 - dones)
            
        loss = nn.MSELoss()(q_values, expected_q_values)
        
        # Epsilon decay
        if self.epsilon > self.hparams.epsilon_end:
            self.epsilon *= self.hparams.epsilon_decay
            
        # Target update
        if self.global_step % 100 == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            
        self.log('train_loss', loss)
        self.log('epsilon', self.epsilon)
        
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
        return [optimizer], [scheduler]

# 訓練腳本
def train_lightning():
    model = DQNLiteModule(state_dim=16, action_dim=4)
    
    trainer = pl.Trainer(
        max_epochs=1, 
        limit_train_batches=5000, 
        gradient_clip_val=1.0, 
        accelerator="auto",
        devices=1
    )
    
    from torch.utils.data import DataLoader, IterableDataset
    
    class RLDataset(IterableDataset):
        def __iter__(self):
            while True:
                yield torch.zeros(1)
                
    loader = DataLoader(RLDataset(), batch_size=1)
    trainer.fit(model, loader)

if __name__ == "__main__":
    print("Starting Training with PyTorch Lightning...")
    train_lightning()
