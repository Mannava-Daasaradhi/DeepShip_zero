import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from src.game.board import Board
from src.agents.alphazero import AlphaZeroAgent
from src.agents.probability import ProbabilityAgent

def train_imitation(episodes=2000, batch_size=512):
    print(f"--- Starting Imitation Learning (Teacher Forcing) ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # 1. Initialize Student (Neural Net) and Teacher (Math Agent)
    student = AlphaZeroAgent(device=device)
    teacher = ProbabilityAgent(device=device)
    
    optimizer = optim.Adam(student.model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    # Data Storage
    states_buffer = []
    actions_buffer = []
    
    print("Step 1: Generating Data (Teacher playing games)...")
    # We generate games in chunks to manage memory
    
    board = Board()
    
    # Validation Metrics
    running_loss = 0.0
    
    # Training Loop
    # We generate a game, then immediately train on it (Online Learning)
    pbar = tqdm(range(episodes))
    
    for e in pbar:
        # --- PLAY GAME (Teacher) ---
        board.place_randomly()
        sunk_ships = []
        
        game_states = []
        game_actions = []
        
        while not board.is_game_over():
            # Teacher decides best move
            # Note: We use the internal heatmap logic directly for speed
            # But here we just ask the agent for the move
            row, col = teacher.get_action(board.state, sunk_ships)
            
            # Record State (Before move)
            # We must use the Student's preprocessor to match input format
            state_tensor = student.preprocess_state(board.state, sunk_ships).cpu()
            
            # Record Action (Target Class)
            action_idx = row * 10 + col
            
            game_states.append(state_tensor)
            game_actions.append(action_idx)
            
            # Execute
            is_hit, is_sunk, ship_name = board.fire(row, col)
            if is_sunk:
                sunk_ships.append(ship_name)
        
        # Add to buffer
        states_buffer.extend(game_states)
        actions_buffer.extend(game_actions)
        
        # --- TRAIN (Student) ---
        # Once we have enough data, do a training step
        if len(states_buffer) >= batch_size:
            # Convert to tensors
            batch_states = torch.cat(states_buffer[:batch_size]).to(device)
            batch_actions = torch.tensor(actions_buffer[:batch_size], dtype=torch.long).to(device)
            
            # Clear used data
            states_buffer = states_buffer[batch_size:]
            actions_buffer = actions_buffer[batch_size:]
            
            # Forward Pass
            # We only care about Policy Head for imitation
            policy_logits, _ = student.model(batch_states)
            
            # Loss: Distance between Student's prediction and Teacher's move
            loss = loss_fn(policy_logits, batch_actions)
            
            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss = 0.9 * running_loss + 0.1 * loss.item()
            pbar.set_description(f"Loss: {running_loss:.4f}")
            
    # Save the educated model
    torch.save(student.model.state_dict(), "models/best_model.pth")
    print("\nImitation Training Complete.")
    print("The Neural Network has now 'cloned' the Math Agent's logic.")

if __name__ == "__main__":
    train_imitation()