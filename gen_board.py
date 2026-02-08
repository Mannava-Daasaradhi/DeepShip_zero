import pickle
import numpy as np
import os
import time
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from src.game.board import Board
from src.agents.probability import ProbabilityAgent

# --- SETTINGS ---
# Use ALL available cores for maximum "Swarm" effect
NUM_WORKERS = os.cpu_count() 
BATCH_UPDATE_FREQ = 50  # Update progress bar every 50 games per worker (reduces lock contention)

def worker_routine(worker_id, target_count, return_dict, progress_queue):
    """
    Independent worker process. 
    Runs full speed, only stopping to report progress periodically.
    """
    # Unique seed per worker prevents identical simulations
    np.random.seed(int(time.time()) + worker_id)
    
    # Initialize Hunter (Try GPU, fall back to CPU if VRAM full)
    try:
        hunter = ProbabilityAgent(device="cuda")
    except:
        hunter = ProbabilityAgent(device="cpu")

    local_results = []
    
    for i in range(target_count):
        board = Board()
        # Pure random placement to explore the widest search space
        board.place_ship_randomly_pure()
        
        # Battle Logic
        sunk_ships = []
        turns = 0
        
        while not board.is_game_over():
            r, c = hunter.get_action(board.state, sunk_ships)
            is_hit, is_sunk, name = board.fire(r, c)
            if is_sunk:
                sunk_ships.append(name)
            turns += 1
            
        # Save minimal config (Name -> Coords)
        layout_config = {}
        for ship in board.ships:
            layout_config[ship.name] = list(ship.coords)
            
        local_results.append((turns, layout_config))
        
        # Report progress in batches to avoid slowing down the CPU
        if (i + 1) % BATCH_UPDATE_FREQ == 0:
            progress_queue.put(BATCH_UPDATE_FREQ)
    
    # Send remaining progress if count isn't divisible by batch size
    remaining = target_count % BATCH_UPDATE_FREQ
    if remaining > 0:
        progress_queue.put(remaining)

    # Store final results
    return_dict[worker_id] = local_results

def get_layout_fingerprint(layout_config):
    """Creates a unique hash for deduplication"""
    processed = []
    for name, coords in layout_config.items():
        sorted_coords = tuple(sorted([tuple(c) for c in coords]))
        processed.append((name, sorted_coords))
    processed.sort(key=lambda x: x[0])
    return tuple(processed)

def generate_survivor_layouts(target_simulations=100000, keep_top=100):
    # Enable CUDA Multiprocessing
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print(f"\n--- EVOLUTION CHAMBER: SWARM MODE ---")
    print(f"Workers: {NUM_WORKERS} parallel cores")
    print(f"Target:  {target_simulations} battles")
    print(f"System:  MAX LIMIT (Progress bar active)")
    
    # Calculate load per worker
    sims_per_worker = target_simulations // NUM_WORKERS
    
    # Setup shared memory objects
    manager = mp.Manager()
    return_dict = manager.dict()
    progress_queue = manager.Queue()
    
    processes = []
    start_time = time.time()
    
    # Launch Swarm
    for i in range(NUM_WORKERS):
        p = mp.Process(target=worker_routine, args=(i, sims_per_worker, return_dict, progress_queue))
        p.start()
        processes.append(p)
    
    # --- MONITOR PROGRESS ---
    # The main process stays here, updating the bar while workers work
    total_reported = 0
    with tqdm(total=target_simulations, unit="games", desc="Simulating") as pbar:
        while True:
            # Check for updates in the queue
            try:
                # Non-blocking check? No, blocking with timeout is better for CPU usage
                inc = progress_queue.get(timeout=0.1)
                pbar.update(inc)
                total_reported += inc
            except:
                # Queue empty (timeout)
                pass

            # Check if we are done
            if total_reported >= (sims_per_worker * NUM_WORKERS):
                break
            
            # Failsafe: Check if all workers died unexpectedly
            if not any(p.is_alive() for p in processes) and progress_queue.empty():
                break

    # Wait for clean exit
    for p in processes:
        p.join()
        
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n--- SWARM RETURNED ({duration:.1f}s) ---")
    print("Filtering Elite Layouts...")
    
    # Aggregation & Deduplication
    all_results = []
    seen_fingerprints = set()
    
    # 1. Load existing (if any) to preserve history
    file_path = "data/best_layouts.pkl"
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                existing = pickle.load(f)
                for layout in existing:
                    fp = get_layout_fingerprint(layout)
                    seen_fingerprints.add(fp)
        except: pass

    # 2. Merge Worker Results
    total_raw = 0
    unique_new = 0
    
    for worker_id, data in return_dict.items():
        total_raw += len(data)
        for score, layout in data:
            fp = get_layout_fingerprint(layout)
            if fp not in seen_fingerprints:
                seen_fingerprints.add(fp)
                all_results.append((score, layout))
                unique_new += 1
    
    # 3. Natural Selection
    all_results.sort(key=lambda x: x[0], reverse=True)
    best_survivors = all_results[:keep_top]
    
    if not best_survivors:
        print("Error: No valid layouts generated.")
        return

    top_score = best_survivors[0][0]
    avg_score = np.mean([x[0] for x in best_survivors])
    min_score = best_survivors[-1][0]
    
    print(f"\n--- CHAMPIONS ---")
    print(f"Top 1 Survivor: {top_score} turns")
    print(f"Top 100 Average: {avg_score:.1f} turns")
    print(f"Weakest Accepted: {min_score} turns")
    print(f"Speed: {total_raw / duration:.0f} games/sec")

    # 4. Save
    os.makedirs("data", exist_ok=True)
    final_save_data = [layout for score, layout in best_survivors]
    
    with open(file_path, "wb") as f:
        pickle.dump(final_save_data, f)
        
    print(f"Saved to '{file_path}'")

if __name__ == "__main__":
    generate_survivor_layouts(target_simulations=100000, keep_top=100)