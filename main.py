import argparse
from src.training.trainer import DeepShipTrainer
from src.game.board import Board
from src.agents.probability import ProbabilityAgent
from src.agents.alphazero import AlphaZeroAgent
import numpy as np

def train():
    trainer = DeepShipTrainer()
    # 5000 games on an RTX 4090 will take about 10-15 minutes
    trainer.run_training(episodes=5000)

def play():
    # Human vs AI
    print("Loading Best Model...")
    try:
        ai = AlphaZeroAgent(model_path="models/best_model.pth")
    except:
        print("No trained model found! Using untrained AI (Expect stupidity).")
        ai = AlphaZeroAgent()
        
    board = Board()
    board.place_randomly()
    print("Game Started! Target Board:")
    
    turns = 0
    sunk_ships = []
    
    while not board.is_game_over():
        # AI Turn
        row, col = ai.get_action(board.state, sunk_ships)
        is_hit, is_sunk, ship_name = board.fire(row, col)
        
        turns += 1
        status = "MISS"
        if is_hit:
            status = "HIT"
            if is_sunk:
                status = f"SUNK ({ship_name})"
                sunk_ships.append(ship_name)
        
        print(f"Turn {turns}: AI fires at {row},{col} -> {status}")
    
    print(f"AI won in {turns} turns.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'play'], help="Mode: 'train' or 'play'")
    args = parser.parse_args()
    
    if args.mode == 'train':
        train()
    else:
        play()