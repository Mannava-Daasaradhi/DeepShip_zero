import pygame
import sys
import numpy as np
import time
import threading
from src.game.board import Board
from src.agents.alphazero import AlphaZeroAgent

# --- CONSTANTS ---
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 700
GRID_SIZE = 40
MARGIN = 2
BOARD_OFFSET_X_HUMAN = 50
BOARD_OFFSET_X_AI = 650
BOARD_OFFSET_Y = 150

# COLORS
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (50, 50, 50)
LIGHT_GRAY = (200, 200, 200)
BLUE = (0, 100, 255)
RED = (255, 50, 50)
GREEN = (50, 200, 50)
YELLOW = (255, 255, 0)
PURPLE = (200, 0, 255)
DARK_BLUE = (0, 0, 50)
SHIP_COLOR = (100, 100, 100)

# INIT PYGAME
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("DeepShip Zero - GOD MODE (MCTS Enabled)")
font = pygame.font.Font(None, 36)
small_font = pygame.font.Font(None, 24)

class DeepShipGUI:
    def __init__(self):
        self.clock = pygame.time.Clock()
        self.human_board = Board()
        self.ai_board = Board()
        self.ai_board.place_randomly()
        
        print("Loading Grandmaster AI...")
        try:
            self.ai_agent = AlphaZeroAgent(model_path="models/best_model.pth")
        except:
            print("Model not found. Using untrained AI.")
            self.ai_agent = AlphaZeroAgent()

        self.state = "SETUP" 
        self.winner = None
        self.ai_sunk_ships = []
        self.ai_thinking = False
        self.ai_move_result = None
        
        self.ships_to_place = list(self.human_board.ship_templates.items())
        self.current_ship_idx = 0
        self.orientation = "H"
        self.message = "Place your ships! Press 'R' to Rotate."

    def draw_grid(self, start_x, start_y, board, is_human_view=True, hide_ships=False):
        label = font.render("YOU" if is_human_view else "AI (TARGET)", True, WHITE)
        screen.blit(label, (start_x + 100, start_y - 40))

        for r in range(10):
            for c in range(10):
                x = start_x + c * (GRID_SIZE + MARGIN)
                y = start_y + r * (GRID_SIZE + MARGIN)
                
                color = BLUE
                cell_value = board.state[r, c]
                
                if cell_value == 1: color = RED 
                elif cell_value == 2: color = WHITE 
                elif cell_value == 3: color = (100, 0, 0)
                
                pygame.draw.rect(screen, color, (x, y, GRID_SIZE, GRID_SIZE))
                
                if not hide_ships and cell_value == 0:
                    if (r, c) in board.ship_map:
                        pygame.draw.rect(screen, SHIP_COLOR, (x, y, GRID_SIZE, GRID_SIZE))

                if cell_value in [1, 3]:
                    start_pos = (x + 5, y + 5)
                    end_pos = (x + GRID_SIZE - 5, y + GRID_SIZE - 5)
                    pygame.draw.line(screen, BLACK, start_pos, end_pos, 3)
                    start_pos = (x + GRID_SIZE - 5, y + 5)
                    end_pos = (x + 5, y + GRID_SIZE - 5)
                    pygame.draw.line(screen, BLACK, start_pos, end_pos, 3)
                elif cell_value == 2:
                    pygame.draw.circle(screen, BLACK, (x + GRID_SIZE//2, y + GRID_SIZE//2), 5)

    def get_grid_coords(self, mouse_x, mouse_y, start_x, start_y):
        if mouse_x < start_x or mouse_y < start_y: return None
        col = (mouse_x - start_x) // (GRID_SIZE + MARGIN)
        row = (mouse_y - start_y) // (GRID_SIZE + MARGIN)
        if 0 <= col < 10 and 0 <= row < 10: return int(row), int(col)
        return None

    def ai_turn_thread(self):
        """Runs the expensive MCTS search in a separate thread."""
        self.ai_thinking = True
        # Use MCTS (Lookahead) instead of simple get_action
        ar, ac = self.ai_agent.get_action_mcts(self.human_board.state, self.ai_sunk_ships)
        self.ai_move_result = (ar, ac)
        self.ai_thinking = False

    def handle_setup(self):
        if self.current_ship_idx < len(self.ships_to_place):
            ship_name, ship_size = self.ships_to_place[self.current_ship_idx]
            info = small_font.render(f"Placing: {ship_name} ({ship_size}) - {self.orientation} (Press 'R' to Rotate)", True, YELLOW)
            screen.blit(info, (50, 100))
            
            mx, my = pygame.mouse.get_pos()
            coords = self.get_grid_coords(mx, my, BOARD_OFFSET_X_HUMAN, BOARD_OFFSET_Y)
            
            if coords:
                r, c = coords
                ghost_coords = []
                valid = True
                for i in range(ship_size):
                    gr, gc = (r, c + i) if self.orientation == "H" else (r + i, c)
                    if not (0 <= gr < 10 and 0 <= gc < 10): valid = False; break
                    if (gr, gc) in self.human_board.ship_map: valid = False
                    ghost_coords.append((gr, gc))
                
                if valid:
                    for gr, gc in ghost_coords:
                        x = BOARD_OFFSET_X_HUMAN + gc * (GRID_SIZE + MARGIN)
                        y = BOARD_OFFSET_Y + gr * (GRID_SIZE + MARGIN)
                        s = pygame.Surface((GRID_SIZE, GRID_SIZE))
                        s.set_alpha(128)
                        s.fill(GREEN)
                        screen.blit(s, (x, y))
                        
                if pygame.mouse.get_pressed()[0] and valid:
                    self.human_board.place_ship_manual(ship_name, r, c, self.orientation)
                    self.current_ship_idx += 1
                    time.sleep(0.2)
        else:
            self.state = "PLAY"
            self.message = "Game Started! Your Turn."

    def run(self):
        running = True
        while running:
            screen.fill(DARK_BLUE)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    self.orientation = "V" if self.orientation == "H" else "H"
                        
                if event.type == pygame.MOUSEBUTTONDOWN and self.state == "PLAY" and not self.ai_thinking:
                    mx, my = pygame.mouse.get_pos()
                    coords = self.get_grid_coords(mx, my, BOARD_OFFSET_X_AI, BOARD_OFFSET_Y)
                    
                    if coords and not self.winner:
                        r, c = coords
                        if self.ai_board.state[r, c] == 0:
                            is_hit, is_sunk, name = self.ai_board.fire(r, c)
                            if is_hit: self.message = f"HIT! You hit {name}!" if not is_sunk else f"SUNK! You sunk {name}!"
                            else: self.message = "MISS."
                                
                            if self.ai_board.is_game_over():
                                self.winner = "HUMAN"
                                self.message = "VICTORY! You defeated the AI."
                            else:
                                # Start AI Turn (Threaded)
                                threading.Thread(target=self.ai_turn_thread).start()

            # Logic Update
            if self.state == "PLAY" and self.ai_move_result:
                ar, ac = self.ai_move_result
                ai_hit, ai_sunk, ai_name = self.human_board.fire(ar, ac)
                if ai_sunk: self.ai_sunk_ships.append(ai_name)
                
                if self.human_board.is_game_over():
                    self.winner = "AI"
                    self.message = "DEFEAT. The AI wins."
                
                self.ai_move_result = None # Reset

            # Draw
            self.draw()
            if self.state == "SETUP": self.handle_setup()
            
            if self.ai_thinking:
                thinking_surf = font.render("AI IS CALCULATING...", True, PURPLE)
                screen.blit(thinking_surf, (SCREEN_WIDTH//2 - 100, 650))

            pygame.display.flip()
            self.clock.tick(30)
            
        pygame.quit()
        sys.exit()

    def draw(self):
        title = font.render("DEEPSHIP ZERO (MCTS)", True, GREEN)
        screen.blit(title, (SCREEN_WIDTH//2 - 120, 20))
        
        msg_color = WHITE if not self.winner else (0, 255, 0)
        msg_surf = font.render(self.message, True, msg_color)
        screen.blit(msg_surf, (SCREEN_WIDTH//2 - msg_surf.get_width()//2, 600))
        
        self.draw_grid(BOARD_OFFSET_X_HUMAN, BOARD_OFFSET_Y, self.human_board, is_human_view=True)
        show_ai = (self.winner is not None)
        self.draw_grid(BOARD_OFFSET_X_AI, BOARD_OFFSET_Y, self.ai_board, is_human_view=False, hide_ships=not show_ai)

if __name__ == "__main__":
    game = DeepShipGUI()
    game.run()