import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from src.game.board import Board
from src.agents.alphazero import AlphaZeroAgent
from tqdm import tqdm
import os

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward):
        self.buffer.append((state, action, reward))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward = zip(*batch)
        return state, action, reward

    def __len__(self):
        return len(self.buffer)

class DeepShipTrainer:
    def __init__(self, learning_rate=0.0001, gamma=0.99): # Lower LR for finetuning
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Finetuning on: {self.device} (RTX 4090 Mode)")
        
        # Load the Imitation Model we just built
        self.agent = AlphaZeroAgent(model_path="models/best_model.pth", device=self.device)
        self.optimizer = optim.Adam(self.agent.model.parameters(), lr=learning_rate)
        
        self.memory = ReplayBuffer(capacity=100000)
        self.batch_size = 1024 
        self.gamma = gamma
        
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()

    def self_play(self, epsilon=0.1):
        board = Board()
        board.place_randomly()
        
        game_history = []
        state = board.state.copy()
        sunk_ships = []
        
        while not board.is_game_over():
            # Get move from AI
            # Epsilon ensures it tries new things occasionally
            row, col = self.agent.get_action(state, sunk_ships, epsilon=epsilon)
            
            state_tensor = self.agent.preprocess_state(state, sunk_ships).cpu()
            action_idx = row * 10 + col
            
            game_history.append({
                'state': state_tensor,
                'action': action_idx
            })
            
            is_hit, is_sunk, ship_name = board.fire(row, col)
            state = board.state.copy()
            if is_sunk:
                sunk_ships.append(ship_name)

        # REWARD FUNCTION V2 (Aggressive)
        turns = board.shots_fired
        
        # We only reward games that are BETTER than the current average (45)
        if turns <= 40:
            final_reward = 1.0   # GODLIKE
        elif turns <= 45:
            final_reward = 0.5   # Good
        elif turns <= 50:
            final_reward = 0.1   # Okay
        else:
            final_reward = -1.0  # Trash (Punish slow games heavily)
            
        return game_history, final_reward

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        states, actions, rewards = self.memory.sample(self.batch_size)

        # Stack tensors (Using cat for states to keep correct shape)
        state_batch = torch.cat(states).to(self.device)
        action_batch = torch.tensor(actions, dtype=torch.long).to(self.device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(1)

        policy_logits, value_pred = self.agent.model(state_batch)

        loss_value = self.value_loss_fn(value_pred, reward_batch)
        loss_policy = self.policy_loss_fn(policy_logits, action_batch)

        total_loss = loss_policy + loss_value

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def run_training(self, episodes=2000):
        print(f"Starting Finetuning for {episodes} episodes...")
        pbar = tqdm(range(episodes))
        
        for e in pbar:
            # Low epsilon because we are already good, just need refinement
            epsilon = max(0.05, 0.2 - (e / episodes))
            
            history, reward = self.self_play(epsilon)
            
            for step in history:
                self.memory.push(step['state'], step['action'], reward)
            
            loss = self.train_step()
            
            # Save if we hit a milestone
            if e % 500 == 0:
                torch.save(self.agent.model.state_dict(), "models/best_model.pth")
                
            pbar.set_description(f"Loss: {loss:.4f} | Last Reward: {reward}")
                
        torch.save(self.agent.model.state_dict(), "models/best_model.pth")
        print("Finetuning Complete.")