import json
import math
import os
import random
import numpy as np
from copy import deepcopy

# Assuming these are in your local directory or environment
from rl_service_offline_buffer_training_ablation import offline_rl_agent # Using the updated Balanced DQN Agent

# --- Configuration ---
DATA_DIR = "data"
MODEL_SAVE_PATH = "models/ablation/offline_rl_model_0_1332.pt"

# Simulation Hyperparameters
TARGET_HUMAN_COUNT = None
TARGET_BOT_COUNT = None 
SIMULATION_STEP_SECONDS = 10 
SESSION_CHUNK_SIZE = int(SIMULATION_STEP_SECONDS * 1000) 
TOTAL_EPISODES = 2000
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 200 #1000 Global steps between target network syncs

best_score = -626.4 #-float("inf")

def update_human_bot_population(human_count=None, bot_count=None):
    global TARGET_HUMAN_COUNT, TARGET_BOT_COUNT, conf
    TARGET_HUMAN_COUNT = random.randint(80, 100) if human_count is None else human_count
    TARGET_BOT_COUNT = random.choice([0, 20, 100, 200, 500, 1000]) if bot_count is None else bot_count

def calculate_reward(user, threat_level):
    """
    Refined reward function for balancing Security vs. UX.
    Weights are normalized so that one 'Happy Human' is roughly 
    equivalent to one 'Caught Bot'.
    """
    reward = 0
    done = False
    
    # Constants for tuning
    MAX_REWARD = 50.0        # Baseline for a 'Perfect' action
    LEAKAGE_PENALTY = -200.0 # Bot got through (Critical Failure)
    FRICTION_BASE = 2.5      # Sensitivity to human annoyance
    
    if user.is_bot:
        if threat_level > user.bot_strength:
            # Bot CAUGHT
            overkill = (threat_level - user.bot_strength)
            reward = MAX_REWARD - (overkill * 2.5) # Penalty for using high level on weak bot
            done = True
            user.reset_session()
        else:
            # Bot SURVIVED / LEAKED
            gap = user.bot_strength - threat_level
            reward = LEAKAGE_PENALTY - (gap * 5.0)
            user.captchas_solved += 1
    else:
        # HUMAN logic
        if threat_level == 0:
            reward = MAX_REWARD # Perfect score for zero friction
        else:
            # Level 1 is annoying, Level 10 is session-ending frustration
            frustration = math.pow(FRICTION_BASE, threat_level)
            reward = (MAX_REWARD / 2.0) - frustration
            
        user.captchas_solved += 1

        # Check if human gave up (Simulated patience)
        if threat_level == 10 or random.random() * (threat_level - 6) > 0.5:
            done = True # Human abandoned site due to high friction

    return reward, done


def load_session_data(data_dir):
    """Loads all human and bot raw data files."""
    human_sessions = []
    bot_sessions = []
    print(f"Loading raw session data from {data_dir}...")
    for filename in os.listdir(data_dir):
        if not filename.endswith('.json'): continue
        filepath = os.path.join(data_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            if filename.startswith('human'):
                human_sessions.append(data)
            elif filename.startswith('bot'):
                bot_sessions.append(data)
        except Exception as e:
            print(f"Warning: Could not load or parse {filename}. Error: {e}")

    print(f"Loaded {len(human_sessions)} human sessions and {len(bot_sessions)} bot sessions.")
    if not human_sessions or not bot_sessions:
        print("Error: Not enough data to run simulation. Please run bot simulator and browse site.")
        exit()
    return human_sessions, bot_sessions


def perturb_mouse_data(movements):
    """Creates a 'new' user by slightly altering existing mouse data."""
    if not movements: return []
    # Add small random noise to each coordinate
    noise_x = random.randint(-5//5, 5//5)
    noise_y = random.randint(-5//5, 5//5)
    return [{**m, 'x': m['x'] + noise_x, 'y': m['y'] + noise_y, 'timestamp': m['timestamp']} for m in movements]


def get_session_chunks(session_data, chunk_size_ms, target_chunks_size=12):
    """Splits a full session's mouse data into 10-second (step) chunks."""
    start_time = session_data['timestamps']['start']
    movements = session_data.get('mouse_movements', [])
    if not movements: return [[]]  # Return one empty chunk

    chunks = []
    current_chunk = []
    chunk_end_time = start_time + chunk_size_ms

    for move in movements:
        if move['timestamp'] <= chunk_end_time:
            current_chunk.append(move)
        else:
            chunks.append(current_chunk)
            if len(chunks) == target_chunks_size:
                return chunks
            # Handle multiple empty steps
            while move['timestamp'] > chunk_end_time + chunk_size_ms:
                chunks.append([])
                if len(chunks) == target_chunks_size:
                    return chunks
                chunk_end_time += chunk_size_ms

            current_chunk = [move]
            chunk_end_time += chunk_size_ms

    chunks.append(current_chunk)  # Add the last chunk
    if len(chunks) < target_chunks_size:
        chunks = chunks + get_session_chunks(session_data, chunk_size_ms, target_chunks_size=target_chunks_size - len(chunks))
    return chunks

class SimulatedUser:
    def __init__(self, base_session_data, is_bot):
        self.is_bot = is_bot
        self.client_id = f"{'bot' if is_bot else 'human'}_{random.randint(10000, 99999)}"
        self.captchas_solved = 0
        self.last_captcha_solved = 0
        self.done = False
        self.last_humanity_score = 0.5
        self.average_humanity_score = 0.5
        self.n_humanity_score = 0
        self.bot_strength = random.randint(0, 9) if is_bot else None

        # Perturb and chunk the data to create a new, unique session
        perturbed_movements = perturb_mouse_data(base_session_data.get('mouse_movements', []))
        perturbed_session = {**base_session_data, 'mouse_movements': perturbed_movements}
        self.session_chunks = get_session_chunks(perturbed_session, SESSION_CHUNK_SIZE)
        self.current_chunk_index = 0

    def get_step_data(self):
        if self.current_chunk_index >= len(self.session_chunks):
            self.done = True
            return None
        chunk = self.session_chunks[self.current_chunk_index]
        self.current_chunk_index += 1
        return chunk
    
    def update_stats(self, h_score):
        if h_score is not None:
            self.last_humanity_score = h_score
            self.average_humanity_score = (self.average_humanity_score * self.n_humanity_score + h_score) / (self.n_humanity_score + 1)
            self.n_humanity_score += 1

    def reset_session(self):
        self.captchas_solved = 0
        self.done = True

# --- Main Simulation ---
try:
    print("--- Starting Balanced Offline Training Simulator ---")
    
    # Pre-loading data
    human_sessions, bot_sessions = load_session_data(DATA_DIR)
    
    offline_rl_agent.load_model(MODEL_SAVE_PATH)
    # atexit.register(lambda: offline_rl_agent.save_model(MODEL_SAVE_PATH))

    global_step = 0
    
    for episode in range(1331, TOTAL_EPISODES):
        update_human_bot_population()
        
        # Initialize active users
        print(f"\nCreating population: {TARGET_HUMAN_COUNT} humans, {TARGET_BOT_COUNT} bots")
        active_users = []
        for _ in range(TARGET_HUMAN_COUNT):
            active_users.append(SimulatedUser(random.choice(human_sessions), is_bot=False))
        for _ in range(TARGET_BOT_COUNT):
            active_users.append(SimulatedUser(random.choice(bot_sessions), is_bot=True))

        random.shuffle(active_users)

        total_steps_in_episode = 0
        
        pending_experiences = {} # To track s_t, a_t while waiting for s_t+1
        episode_reward = 0
        episode_h_step_counter = 0
        episode_b_step_counter = 0
        episode_human_reward = 0
        episode_bot_reward = 0

        while active_users:
            server_total = len(active_users)
            next_active_users = []
            
            for user in active_users:                              
                # 2. Current State S_t
                s_t = [
                    user.captchas_solved,
                    user.last_captcha_solved,
                    server_total
                ]
                
                # 3. Handle experience from the PREVIOUS step (Transition S_prev -> S_t)
                if user.client_id in pending_experiences:
                    prev_s, prev_a, prev_r, prev_done = pending_experiences[user.client_id]
                    # We now have the 'next_state' (which is current s_t)
                    offline_rl_agent.remember(prev_s, prev_a, prev_r, s_t, prev_done, user.is_bot)
                
                # 4. Take New Action A_t
                a_t = offline_rl_agent.select_action(s_t)
                
                # 5. Environment Feedback R_t
                r_t, done_t = calculate_reward(user, a_t)
                episode_reward += r_t

                if user.is_bot:
                    episode_bot_reward += r_t
                    episode_b_step_counter += 1
                else:
                    episode_human_reward += r_t
                    episode_h_step_counter += 1
                
                # 6. Store terminal or pending
                if done_t or user.done:
                    # If finished, we store immediately with current state as next_state
                    offline_rl_agent.remember(s_t, a_t, r_t, s_t, True, user.is_bot)
                else:
                    # Still active, wait for next chunk to get next_state
                    pending_experiences[user.client_id] = (s_t, a_t, r_t, False)
                    user.last_captcha_solved = a_t
                    next_active_users.append(user)
                
                # 7. Training Update
                global_step += 1
                if global_step % 5 == 0:
                    offline_rl_agent.train_model(BATCH_SIZE)
                
                if global_step % TARGET_UPDATE_FREQUENCY == 0:
                    offline_rl_agent.update_target_net()

            active_users = next_active_users

        # Episode Summary
        print(f"Episode {episode+1} | Reward: {episode_reward:.2f} | Epsilon: {offline_rl_agent.epsilon:.3f}")
        # Metrics logging...
        avg_rew = episode_reward / (episode_h_step_counter + episode_b_step_counter)   # (TARGET_HUMAN_COUNT + TARGET_BOT_COUNT)
        avg_human_rew = episode_human_reward / episode_h_step_counter
        avg_bot_rew = episode_bot_reward / episode_b_step_counter if episode_b_step_counter else 0
        print(f"Total Reward: {episode_reward:.2f} | Avg: {avg_rew:.2f} | Avg Human: {avg_human_rew:.2f} | Avg Bot: {avg_bot_rew:.2f}")
        
        avg_session_rew = episode_reward / (TARGET_HUMAN_COUNT + TARGET_BOT_COUNT)
        avg_h_session_rew = episode_human_reward / (TARGET_HUMAN_COUNT)
        avg_b_session_rew = episode_bot_reward / TARGET_BOT_COUNT if TARGET_BOT_COUNT else 0
        print(f"Average Session Rewards::: Mean: {avg_session_rew:.2f} | Human: {avg_h_session_rew:.2f} | Bot: {avg_b_session_rew:.2f}")

        EPISODE_REWARD = {
            "reward": episode_reward,
            "human_reward": episode_human_reward,
            "bot_reward": episode_bot_reward,
            "human_steps": episode_h_step_counter,
            "bot_steps": episode_b_step_counter,
            "epsilon": offline_rl_agent.epsilon,
            "human_population": TARGET_HUMAN_COUNT,
            "bot_population": TARGET_BOT_COUNT,
            "population": TARGET_HUMAN_COUNT + TARGET_BOT_COUNT,
        }

        with open("models/ablation/metrics/metrics.jsonl", "a") as f:
            f.write(json.dumps(EPISODE_REWARD) + "\n")

        if offline_rl_agent.epsilon > 0.1:
            continue

        if (episode + 1) % 50 == 0:
            offline_rl_agent.save_model(f"models/ablation/offline_rl_model_0_{episode+1}.pt")

        cur_score = (episode_human_reward / episode_h_step_counter) + (episode_bot_reward / episode_b_step_counter if episode_b_step_counter else 0)
        if episode_b_step_counter > 0 and cur_score > best_score:
            best_score = cur_score
            offline_rl_agent.save_model(filepath=f"models/ablation/best_model_{best_score:.2f}.pt")
            print(f"Saved Best Model with Score = {best_score:.2f}!")

except Exception | KeyboardInterrupt as e:
    print("Shutting down...")
    print(e)

finally:
    offline_rl_agent.save_model(f"models/ablation/offline_rl_model_0_{episode+1}.pt")
