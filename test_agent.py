import numpy as np
from src.game.board import Board
from src.agents.probability import ProbabilityAgent

def test_game():
    # Initialize
    board = Board()
    board.place_randomly() # Place ships for the "Computer"
    
    agent = ProbabilityAgent()
    
    turns = 0
    print("Starting Game with Probability Agent...")
    
    while not board.is_game_over():
        # Get AI move
        # We need to pass the list of sunk ships names
        sunk_names = [s.name for s in board.ships if s.is_sunk]
        
        row, col = agent.get_action(board.state, sunk_names)
        
        # Fire
        is_hit, is_sunk, ship_name = board.fire(row, col)
        
        turns += 1
        if is_hit:
            status = f"HIT! ({ship_name})" if not is_sunk else f"SUNK! ({ship_name})"
            print(f"Turn {turns}: Fired at ({row}, {col}) -> {status}")
    
    print(f"Game Over! Total Turns: {turns}")
    # A random agent takes ~95 turns. 
    # This Math agent should take ~40-60 turns.

if __name__ == "__main__":
    test_game()