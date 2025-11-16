import json
import os
import random
import time
import requests
import numpy as np
from datetime import datetime
import atexit

# Import the new offline agent
from rl_service_offline import offline_rl_agent

# --- Configuration ---
DATA_DIR = "data"
HUMANITY_SCORE_PREDICTOR_URL = "http://127.0.0.1:8000/predict"
MODEL_SAVE_PATH = "models/offline_rl_model.pth"

# Simulation Hyperparameters
TARGET_HUMAN_COUNT = 100  # Number of "real" human users to simulate
TARGET_BOT_COUNT = 80  # Number of bot users to simulate (for DoS)
SIMULATION_STEP_SECONDS = 10  # How often the RL agent is called (matches frontend)
SESSION_CHUNK_SIZE = int(SIMULATION_STEP_SECONDS * 1000)  # In milliseconds
TOTAL_EPISODES = 0
BATCH_SIZE = 16
TARGET_UPDATE_FREQUENCY = 400  # Update target network every 10 steps


# --- Utility Functions ---

def get_humanity_score(mouse_movement):
    """Calls the external humanity score ML model."""
    if not mouse_movement: return 0.5
    # Format for the ML model
    formatted_movements = [[m['x'], m['y']] for m in mouse_movement]
    payload = {"mouse_movement": formatted_movements}
    try:
        response = requests.post(HUMANITY_SCORE_PREDICTOR_URL, json=payload, timeout=2)
        if response.status_code == 200:
            prediction = response.json().get("prediction")
            return prediction if prediction is not None else 0.5
        else:
            print(f"Humanity score predictor error: {response.status_code}")
            return 0.5
    except requests.exceptions.RequestException as e:
        print(f"Humanity score predictor connection error: {e}")
        return 0.5  # Return neutral score on error


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
    noise_x = random.randint(-5, 5)
    noise_y = random.randint(-5, 5)
    return [{**m, 'x': m['x'] + noise_x, 'y': m['y'] + noise_y, 'timestamp': m['timestamp']} for m in movements]


def get_session_chunks(session_data, chunk_size_ms):
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
            # Handle multiple empty steps
            while move['timestamp'] > chunk_end_time + chunk_size_ms:
                chunks.append([])
                chunk_end_time += chunk_size_ms

            current_chunk = [move]
            chunk_end_time += chunk_size_ms

    chunks.append(current_chunk)  # Add the last chunk
    return chunks


class SimulatedUser:
    """Represents a single concurrent user (bot or human) in the simulation."""

    def __init__(self, base_session_data, is_bot):
        self.is_bot = is_bot
        self.client_id = f"{'bot' if is_bot else 'human'}_{random.randint(10000, 99999)}"
        self.captchas_solved = 0
        self.last_captcha_solved = -1
        self.done = False  # Is this user's session over?

        # *** NEW: Assign a "strength" to each bot ***
        # This is the max threat level (0-9) the bot can "beat"
        # A threat level of 10 will *always* kill the bot.
        if self.is_bot:
            self.bot_strength = random.randint(0, 9)
        else:
            self.bot_strength = None  # Humans don't have a strength

        # Perturb and chunk the data to create a new, unique session
        perturbed_movements = perturb_mouse_data(base_session_data.get('mouse_movements', []))
        perturbed_session = {**base_session_data, 'mouse_movements': perturbed_movements}
        self.session_chunks = get_session_chunks(perturbed_session, SESSION_CHUNK_SIZE)
        self.current_chunk_index = 0

    def get_step_data(self):
        """Gets the mouse data for the current 10-second step."""
        if self.done or self.current_chunk_index >= len(self.session_chunks):
            self.done = True
            return None  # This user has finished their session

        chunk = self.session_chunks[self.current_chunk_index]
        self.current_chunk_index += 1
        return chunk

    def reset_session(self):
        """Called when a bot is 'killed'."""
        self.captchas_solved = 0
        self.done = True  # For simulation, a 'kill' ends the session


# --- Graceful Shutdown ---
def shutdown_hook():
    """Save the model on exit."""
    print("\nShutting down... Saving model...")
    offline_rl_agent.save_model(MODEL_SAVE_PATH)
    print("Model saved. Goodbye.")


# --- Main Training Loop ---

def calculate_reward(user, threat_level):
    """
    Calculates reward and done status based on the new deterministic "strength" model.
    """
    reward = 0
    done = False

    if user.is_bot:
        # *** NEW: Deterministic check against bot's strength ***
        if threat_level > user.bot_strength:
            reward = 50.0 - (threat_level - user.bot_strength) * 5   # Caught bot
            done = True
            user.reset_session()
        else:
            reward = -10.0 - (user.bot_strength - threat_level)  # Bot survived
            user.captchas_solved += 1
    else:
        # Human reward is the same
        reward = 5 - threat_level  # Human experience reward
        user.captchas_solved += 1

    return reward, done


def run_offline_training():
    print("--- Starting Offline Training Simulator ---")
    base_human_sessions, base_bot_sessions = load_session_data(DATA_DIR)

    # Load the agent's progress (if any)
    offline_rl_agent.load_model(MODEL_SAVE_PATH)

    # Register the shutdown hook
    atexit.register(shutdown_hook)

    global_step = 0
    timesteps_without_training_tgt_net = 0
    start_time = time.time()

    for episode in range(TOTAL_EPISODES):
        print(f"\n--- Starting Episode {episode + 1}/{TOTAL_EPISODES} ---")

        # 1. Create the population for this episode
        print(f"Creating population: {TARGET_HUMAN_COUNT} humans, {TARGET_BOT_COUNT} bots")
        active_users = []
        for _ in range(TARGET_HUMAN_COUNT):
            active_users.append(SimulatedUser(random.choice(base_human_sessions), is_bot=False))
        for _ in range(TARGET_BOT_COUNT):
            active_users.append(SimulatedUser(random.choice(base_bot_sessions), is_bot=True))

        random.shuffle(active_users)

        total_steps_in_episode = 0
        episode_reward = 0

        # This dict holds the (s, a, r, done) tuple from step t-1, waiting for s' at step t
        pending_experiences = {}

        # 2. Run the simulation step-by-step
        while active_users:
            total_steps_in_episode += 1
            global_step += len(active_users)
            timesteps_without_training_tgt_net += len(active_users)

            # --- Global State Calculation (for step t) ---
            server_total_users = len(active_users)
            server_req_rate_per_min = server_total_users * 7   # (random.randint(5, 15))

            next_active_users = []
            new_pending_experiences = {}

            for user in active_users:
                # 1. Get user's data and calculate current state s_t
                mouse_data = user.get_step_data()
                humanity_score = get_humanity_score(mouse_data)
                client_req_rate_per_min = 6 + random.randint(-2, 2)

                s_t = [
                    humanity_score,
                    user.captchas_solved,
                    user.last_captcha_solved, # client_req_rate_per_min,
                    server_req_rate_per_min,
                    server_total_users
                ]

                # 2. Finalize the experience from step t-1
                if user.client_id in pending_experiences:
                    prev_s, prev_a, prev_r, prev_done = pending_experiences[user.client_id]
                    s_prime = s_t  # The current state is the "next_state" for the previous action

                    # Store the *completed* transition from t-1
                    offline_rl_agent.remember(prev_s, prev_a, prev_r, s_prime, prev_done)
                    episode_reward += prev_r

                # 3. Check if user's session ended *naturally* (ran out of data)
                if user.done:
                    continue  # User is done, no new action to take

                # 4. User is still active, take a new action (a_t)
                a_t = offline_rl_agent.select_action(s_t)

                # 5. Calculate reward (r_t) and if the action *caused* the session to end (done_t)
                r_t, done_t = calculate_reward(user, a_t)

                # print(f"{'Bottt' if user.is_bot else 'Human'} -> action: {a_t}, reward {r_t}, done: {done_t}, score: {humanity_score}")

                # 6. Store this new experience
                if done_t:
                    # This is a terminal state (e.g., bot was killed).
                    # We have the full experience now. Store it immediately.
                    s_prime = s_t  # The 'next_state' is arbitrary, as 'done=True' nullifies it.
                    offline_rl_agent.remember(s_t, a_t, r_t, s_prime, done_t)
                    episode_reward += r_t
                else:
                    # This is a non-terminal state.
                    # Store it as pending, waiting for s' from the next step.
                    new_pending_experiences[user.client_id] = (s_t, a_t, r_t, done_t)
                    user.last_captcha_solved = a_t
                    next_active_users.append(user)  # User continues

            # --- End of Step ---
            active_users = next_active_users
            pending_experiences = new_pending_experiences

            # Train the model in batches
            if len(offline_rl_agent.memory) > BATCH_SIZE * 2:  # Wait for a decent buffer
                print(f"Step {global_step}: Updating behaviour network...")
                offline_rl_agent.train_model(BATCH_SIZE)

            # Update the target network periodically
            # if global_step % TARGET_UPDATE_FREQUENCY == 0:
            if timesteps_without_training_tgt_net >= TARGET_UPDATE_FREQUENCY:
                print(f"Step {global_step}: Updating target network...")
                offline_rl_agent.update_target_net()
                timesteps_without_training_tgt_net -= TARGET_UPDATE_FREQUENCY

            if not active_users:
                print(f"All users finished for episode {episode + 1}.")

        # --- End of Episode ---
        # Flush any remaining pending experiences from users who finished naturally
        for client_id, (s, a, r, done) in pending_experiences.items():
            # This was the last action. 'done' is False, but the *session* is done.
            # This is a terminal state for the *episode*.
            offline_rl_agent.remember(s, a, r, s, True)  # Store as terminal
            episode_reward += r

        print(f"Episode {episode + 1} finished in {total_steps_in_episode} steps. Total Reward: {episode_reward:.2f}")
        print(f"Current Epsilon: {offline_rl_agent.epsilon:.4f}")

    # --- End of Training ---
    end_time = time.time()
    print(f"\n--- Offline Training Complete ---")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Total global steps: {global_step}")

    # The shutdown hook will save the final model


if __name__ == "__main__":
    run_offline_training()
