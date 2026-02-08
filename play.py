import torch
import numpy as np
import os
from src.game.board import Board
from src.agents.alphazero import AlphaZeroAgent

# Colors for terminal
RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_boards(human_board, ai_board, ai_last_prob=None):
    os.system('clear' if os.name == 'posix' else 'cls')
    print(f"{BOLD}      YOUR BOARD (AI Target)                  AI BOARD (Your Target){RESET}")
    print("   0 1 2 3 4 5 6 7 8 9           0 1 2 3 4 5 6 7 8 9")
    
    # 0=Empty, 1=Hit, 2=Miss, 3=Sunk part
    symbols = {0: '.', 1: f'{RED}X{RESET}', 2: f'{BLUE}O{RESET}', 3: f'{RED}#{RESET}'}
    
    for r in range(10):
        # Human Row
        h_row = f"{r} "
        for c in range(10):
            cell = human_board.state[r, c]
            sym = symbols.get(cell, '?')
            # If it's a ship part (not hit yet), show it as S
            if (r, c) in human_board.ship_map and cell == 0:
                sym = f'{GREEN}S{RESET}'
            h_row += f" {sym}"
            
        # AI Row (Hidden)
        a_row = f"  {r} "
        for c in range(10):
            cell = ai_board.state[r, c]
            # Fog of War: Hide ships (0)
            sym = symbols.get(cell, '?')
            if cell == 0: sym = '.' 
            a_row += f" {sym}"
            
        print(h_row + "       " + a_row)

    if ai_last_prob is not None:
        print(f"\nAI Confidence on last shot: {YELLOW}{ai_last_prob:.1%}{RESET}")

def get_human_move(board):
    while True:
        try:
            move = input(f"\n{BOLD}Enter coords to fire (row col): {RESET}")
            r, c = map(int, move.split())
            if 0 <= r < 10 and 0 <= c < 10:
                # Check if already fired
                if board.state[r, c] == 0: # 0 is the only valid target for us (we don't see ships)
                    return r, c
                elif board.state[r, c] in [1, 2, 3]:
                     print("You already fired there!")
                else:
                    # Should not happen given Fog of War logic above, 
                    # but technically we only fire at 0 (Unknown)
                    return r, c
            else:
                print("Out of bounds (0-9).")
        except ValueError:
            print("Invalid format. Use: 3 4")

def play():
    # Load the Grandmaster
    try:
        ai_agent = AlphaZeroAgent(model_path="models/best_model.pth")
        print(f"{GREEN}Loaded Grandmaster AI Model.{RESET}")
    except:
        print(f"{RED}CRITICAL: Model not found. AI will be dumb.{RESET}")
        ai_agent = AlphaZeroAgent()

    # Setup Boards
    human_board = Board()
    ai_board = Board()
    
    # Place Ships
    print("Auto-placing ships for fair start...")
    human_board.place_randomly()
    ai_board.place_randomly()
    
    game_over = False
    ai_sunk_ships = []
    
    while not game_over:
        # --- UI UPDATE ---
        print_boards(human_board, ai_board)
        
        # --- HUMAN TURN ---
        h_r, h_c = get_human_move(ai_board)
        is_hit, is_sunk, name = ai_board.fire(h_r, h_c)
        msg = "MISS"
        if is_hit: msg = "HIT"
        if is_sunk: msg = f"SUNK {name}!"
        print(f"You fired at {h_r},{h_c}: {BOLD}{msg}{RESET}")
        
        if ai_board.is_game_over():
            print(f"\n{GREEN}{BOLD}YOU WIN! The AI has been defeated.{RESET}")
            break
            
        # --- AI TURN ---
        # 1. Get Probability Heatmap (To show confidence)
        # We peek into the brain
        heatmap = ai_agent.math_agent.generate_heatmap(human_board.state, ai_sunk_ships)
        if hasattr(heatmap, "cpu"): heatmap = heatmap.cpu().numpy()
        
        # 2. Get AI Move
        a_r, a_c = ai_agent.get_action(human_board.state, ai_sunk_ships)
        
        # 3. Get confidence at that spot
        confidence = 0.0
        if np.max(heatmap) > 0:
            confidence = heatmap[a_r, a_c] / np.max(heatmap)
            
        # 4. Fire
        is_hit, is_sunk, name = human_board.fire(a_r, a_c)
        if is_sunk:
            ai_sunk_ships.append(name)
            
        print_boards(human_board, ai_board, confidence)
        print(f"AI fired at {a_r},{a_c}.")
        
        if human_board.is_game_over():
            print(f"\n{RED}{BOLD}GAME OVER. The AI wins.{RESET}")
            break

if __name__ == "__main__":
    play()