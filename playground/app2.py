from flask import Flask, request, jsonify, render_template
import json
import os
import time
from datetime import datetime
import random
import requests
import atexit

# Import the RL agent from our new service
# The agent is still needed to *select actions*, even if we're not training it.
from rl_service import rl_agent

# --- Global Configuration ---
# SET THIS TO True FOR ONLINE LEARNING, False FOR DATA COLLECTION
TRAINING_MODE = False

HUMANITY_SCORE_PREDICTOR_URL = "http://127.0.0.1:8000/predict"
MODEL_SAVE_PATH = "models/rl_model.pth"
RL_BATCH_SIZE = 4

# --- In-Memory Logs ---
REWARD_MEMORY = []  # For simple reward-over-time metrics
client_sessions = {}  # For tracking captchas_solved per client

# Logs for Offline Data Collection Mode (if TRAINING_MODE = False)
OFFLINE_TRAINING_LOG = []  # Logs the (s, a, r, s', d) tuples for offline RL
OFFLINE_ANALYSIS_LOG = []  # Logs human-readable data for analysis


def get_humanity_score(mouse_movement):
    if not mouse_movement: return 0.5  # Return a neutral score if no movement data
    mouse_movement = [[movement['x'], movement['y']] for movement in mouse_movement]
    payload = {"mouse_movement": mouse_movement}
    try:
        # Make POST request
        response = requests.post(HUMANITY_SCORE_PREDICTOR_URL, json=payload, timeout=2)

        # Print response
        if response.status_code == 200:
            prediction = response.json().get("prediction")
            print("Prediction:", prediction)
            return prediction if prediction is not None else 0.5
        else:
            print("Error from humanity score predictor:", response.status_code, response.text)
            return 0.5  # Return neutral score on error
    except requests.exceptions.RequestException as e:
        print(f"Could not connect to humanity score predictor: {e}")
        return 0.5  # Return neutral score on connection error


def format_timestamp(ms_timestamp):
    dt_local = datetime.fromtimestamp(ms_timestamp / 1000.0)
    formatted_time = dt_local.strftime('%Y-%m-%d_%H-%M-%S')
    return formatted_time


app = Flask(__name__)

# --- Directory Setup ---
DATA_DIR = 'data'
BLOGS_DIR = 'blogs'
MODELS_DIR = 'models'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(BLOGS_DIR):
    os.makedirs(BLOGS_DIR)
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

if not os.path.exists(os.path.join(BLOGS_DIR, 'blog1.json')):
    # Create a sample blog post if the directory is new
    sample_blog = {
        "id": 1, "topic": "Technology", "heading": "The Rise of Adaptive Security", "author": "Jane Doe",
        "date": "2025-08-15",
        "content": "<p>The landscape of cybersecurity is in a constant state of flux. Traditional, static defenses are no longer sufficient to counter the sophisticated, automated threats that emerge daily.</p><p>This is where adaptive security comes into play. By leveraging machine learning and real-time data analysis, these systems can dynamically adjust their posture to identify and neutralize threats as they happen. Our project is an exploration into this exciting and critical field, aiming to build a CAPTCHA system that learns and evolves.</p>"
    }
    with open(os.path.join(BLOGS_DIR, 'blog1.json'), 'w') as f:
        json.dump(sample_blog, f, indent=4)


# --- Frontend Route ---
@app.route('/')
def playground():
    """Serves the main playground website."""
    return render_template('index.html')


# --- API Endpoints ---
@app.route('/api/blogs')
def get_blogs():
    """Reads all blog JSON files and returns them as a list."""
    blogs = []
    try:
        for filename in sorted(os.listdir(BLOGS_DIR)):
            if filename.endswith('.json'):
                filepath = os.path.join(BLOGS_DIR, filename)
                with open(filepath, 'r') as f:
                    blogs.append(json.load(f))
        return jsonify(blogs)
    except Exception as e:
        print(f"Error reading blogs: {e}")
        return jsonify({"status": "error", "message": "Could not load blog posts"}), 500


@app.route('/api/data_stream', methods=['POST'])
def data_stream():
    """
    Receives periodic data, calculates humanity, gets a threat level from the RL agent.
    Based on TRAINING_MODE, either trains online or logs data for offline training.
    """
    data = request.get_json()
    bot_info = data.get('bot_info')
    is_bot = bot_info is not None

    # Use bot_id for bots, or session start time for humans as a unique identifier
    client_id = bot_info['bot_id'] if is_bot else data['timestamps']['start']

    # Retrieve or initialize the client's session
    if client_id not in client_sessions:
        client_sessions[client_id] = {'captchas_solved': 0}

    session = client_sessions[client_id]

    # --- 1. Humanity Score Calculation ---
    humanity_score = get_humanity_score(mouse_movement=data.get('mouse_movements', None))
    print("Humanity Score: ", humanity_score)

    # --- 2. State Representation ---
    state = [humanity_score, min(10, session['captchas_solved'])]

    # --- 3. RL Agent Action ---
    threat_level = rl_agent.select_action(state)

    # --- 4. Reward Calculation & Simulation (This logic must run in *both* modes) ---
    reward = 0
    done = False
    session_reset = False
    client_type = "human"

    if is_bot:
        client_type = "bot"
        if random.random() < (threat_level / 10.0):
            reward = 10.0
            print(f"Bot {client_id} caught! Threat: {threat_level}/10. REWARD: {reward}. Resetting session.")
            done = True
            session_reset = True
            session['captchas_solved'] = 0  # Reset the bot's state
            if client_id in client_sessions:
                del client_sessions[client_id]  # Remove client id from sessions
        else:
            reward = -3.0
            print(f"Bot {client_id} survived. Threat: {threat_level}/10. PUNISHMENT: {reward}")
            session['captchas_solved'] += 1  # Bot survived, increment counter
    else:
        reward = 5 - threat_level
        print(f"Human user. Threat: {threat_level}/10. Reward: {reward}")
        session['captchas_solved'] += 1

    # --- 5. Handle Training or Data Logging ---
    next_state = [humanity_score, min(10, session.get('captchas_solved', 0))]

    if TRAINING_MODE:
        # ONLINE MODE: Remember and train the model immediately
        rl_agent.remember(state, threat_level, reward, next_state, done)
        rl_agent.train_model(batch_size=RL_BATCH_SIZE)
    else:
        # OFFLINE MODE: Log the experience for later training
        OFFLINE_TRAINING_LOG.append((state, threat_level, reward, next_state, done))

        # Log the detailed analysis data you requested
        analysis_entry = {
            "timestamp": int(time.time() * 1000),
            "client_id": client_id,
            "client_type": client_type,
            "bot_name": bot_info['bot_name'] if is_bot else None,
            "humanity_score": humanity_score,
            "state": state,
            "action_taken": int(threat_level),
            "reward_received": reward,
            "next_state": next_state,
            "is_terminal_step": done
        }
        OFFLINE_ANALYSIS_LOG.append(analysis_entry)

    # Always log simple reward metrics
    REWARD_MEMORY.append((client_type, reward))

    return jsonify({
        "status": "processed",
        "threat_level": int(threat_level),
        "humanity_score": humanity_score,
        "session_reset": session_reset
    })


@app.route('/api/collect', methods=['POST'])
def collect_data():
    """Receives final user interaction data and saves it to a file."""
    try:
        data = request.get_json()
        if not data or 'timestamps' not in data or 'start' not in data['timestamps']:
            return jsonify({"status": "error", "message": "Invalid data format"}), 400

        if data.get('bot_info', None):
            timestamp = data.get('bot_info', {}).get('session_timestamp', data['timestamps']['start'])
            formatted_timestamp = format_timestamp(timestamp)
            bot_name = data.get('bot_info', {}).get('bot_name', 'UnknownBot')
            filename = f"bot_{bot_name}_data_{formatted_timestamp}.json"
        else:
            timestamp = data['timestamps'].get('start', int(time.time() * 1000))
            formatted_timestamp = format_timestamp(timestamp)
            filename = f"human_data_{formatted_timestamp}.json"

        filepath = os.path.join(DATA_DIR, filename)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

        return jsonify({"status": "success", "message": f"Data saved to {filename}"}), 200
    except Exception as e:
        print(f"Error collecting data: {e}")
        return jsonify({"status": "error", "message": "An internal error occurred"}), 500


# --- Graceful Shutdown ---
def shutdown_hook():
    """
    This function is called when the application is shutting down.
    It saves the appropriate model or data based on the TRAINING_MODE.
    """
    if not REWARD_MEMORY:
        print("No activity detected. Shutting down without saving.")
        return

    print("Shutting down gracefully...")
    cur_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # --- Save Reward Metrics (Always) ---
    reward_metric_filename = f"reward_metric_{cur_time}.json"
    with open(reward_metric_filename, "w") as f:
        json.dump(REWARD_MEMORY, f, indent=4)
    print(f"Reward metrics saved to file {reward_metric_filename}")

    if TRAINING_MODE:
        # --- ONLINE MODE: Save the trained model ---
        print("Saving online-trained RL model...")
        rl_agent.save_model(MODEL_SAVE_PATH)
    else:
        # --- OFFLINE MODE: Save the collected data logs ---
        print("Saving offline data logs...")

        # 1. Save the RL training dataset
        offline_rl_filename = f"offline_rl_dataset_{cur_time}.json"
        with open(offline_rl_filename, "w") as f:
            json.dump(OFFLINE_TRAINING_LOG, f, indent=4)
        print(f"Offline RL training dataset saved to {offline_rl_filename}")

        # 2. Save the analysis log
        offline_analysis_filename = f"offline_analysis_log_{cur_time}.json"
        with open(offline_analysis_filename, "w") as f:
            json.dump(OFFLINE_ANALYSIS_LOG, f, indent=4)
        print(f"Offline analysis log saved to {offline_analysis_filename}")


if __name__ == '__main__':
    # Load the RL model on startup.
    # We always need the model to *select actions*, even in data collection mode.
    rl_agent.load_model(MODEL_SAVE_PATH)

    # Register the shutdown hook to save the model or data on exit
    atexit.register(shutdown_hook)

    # Run the Flask app
    app.run(debug=True, host="0.0.0.0", port=5000)

