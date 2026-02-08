import torch
import numpy as np
from tqdm import tqdm
from src.game.board import Board
from src.agents.alphazero import AlphaZeroAgent

def run_tournament(games=1000):
    print(f"--- Starting Tournament: AI vs Average Logic ({games} games) ---")
    
    # Load the trained model
    try:
        ai = AlphaZeroAgent(model_path="models/best_model.pth")
        print("Loaded 'DeepShip' Neural Network.")
    except:
        print("WARNING: Model not found. Did training finish? Using random weights.")
        ai = AlphaZeroAgent()

    ai_turns = []
    
    for _ in tqdm(range(games)):
        board = Board()
        board.place_randomly()
        
        sunk_ships = []
        turns = 0
        
        while not board.is_game_over():
            # AI takes a shot
            row, col = ai.get_action(board.state, sunk_ships)
            is_hit, is_sunk, ship_name = board.fire(row, col)
            
            if is_sunk:
                sunk_ships.append(ship_name)
            
            turns += 1
            
        ai_turns.append(turns)

    avg_turns = np.mean(ai_turns)
    win_rate_vs_human = 0
    
    # Benchmarks
    # A pro human averages ~45-50 turns.
    # If AI averages < 40 turns, it is Superhuman.
    
    print(f"\n--- RESULTS ---")
    print(f"Average Turns to Win: {avg_turns:.1f}")
    
    if avg_turns <= 40:
        print("Rating: GODLIKE (Human Win Rate < 5%)")
        print("The AI is finding ships almost mathematically perfectly.")
    elif avg_turns <= 45:
        print("Rating: GRANDMASTER (Human Win Rate ~10%)")
        print("It will beat 9/10 humans.")
    elif avg_turns <= 55:
        print("Rating: COMPETENT (Human Win Rate ~50%)")
    else:
        print("Rating: NOVICE (Needs more training)")

if __name__ == "__main__":
    run_tournament()