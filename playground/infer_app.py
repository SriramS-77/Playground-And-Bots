from flask import Flask, request, jsonify, render_template
import json
import os
import time
from datetime import datetime
import random
import requests
from rl_service import rl_agent
import atexit
import rl_service_offline


COLLECT_DATA = False
DATA_DIR = 'test_data'
HUMANITY_SCORE_PREDICTOR_URL = "http://127.0.0.1:8000/predict"
MODEL_SAVE_PATH = "models/rl_model.pth"
RL_BATCH_SIZE = 4

REWARD_MEMORY = []

from rl_service_offline import offline_rl_agent

offline_rl_agent.load_model("models/offline_rl_model.pt")


def get_humanity_score(mouse_movement):
    if not mouse_movement: return 0.5   # Return a neutral score if no movement data
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
            return 0.5   # Return neutral score on error
    except requests.exceptions.RequestException as e:
        print(f"Could not connect to humanity score predictor: {e}")
        return 0.5   # Return neutral score on connection error

def format_timestamp(ms_timestamp):
    dt_local = datetime.fromtimestamp(ms_timestamp / 1000.0)
    formatted_time = dt_local.strftime('%Y-%m-%d_%H-%M-%S')
    return formatted_time

app = Flask(__name__)

# --- In-Memory Session Storage ---
# In a production system, this would be a database like Redis.
client_sessions = {}

# --- Directory Setup ---
BLOGS_DIR = 'blogs'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(BLOGS_DIR):
    os.makedirs(BLOGS_DIR)
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
    Receives periodic data, calculates humanity, gets a threat level from the RL agent,
    calculates reward, and trains the agent.
    """
    global offline_rl_agent
    
    data = request.get_json()
    bot_info = data.get('bot_info')
    is_bot = bot_info is not None

    # Use bot_id for bots, or session start time for humans as a unique identifier
    client_id = bot_info['bot_id'] if is_bot else data['timestamps']['start']

    # Retrieve or initialize the client's session
    if client_id not in client_sessions:
        client_sessions[client_id] = {'captchas_solved': 0, 'last_captcha_solved': -1}
        if is_bot:
            client_sessions[client_id]['bot_strength'] = random.randint(0, 9)

    session = client_sessions[client_id]

    # --- 1. Humanity Score Calculation (Placeholder) ---
    # mouse_activity = len(data.get('mouse_movements', []))
    # humanity_score = min(1.0, round(mouse_activity / 200.0, 1))
    humanity_score = get_humanity_score(mouse_movement=data.get('mouse_movements', None))
    print("Humaity Score: ", humanity_score)

    # --- 2. State Representation ---
    state = [
        humanity_score,
        session['captchas_solved'],
        session['last_captcha_solved'],
        180 * 7,
        180
    ]
    # state = [humanity_score, min(10, session['captchas_solved'])]

    # --- 3. RL Agent Action ---
    a_t = offline_rl_agent.select_action(state)
    threat_level = a_t   # rl_agent.select_action(state)

    # --- 4. Reward Calculation & Simulation ---
    done = False
    session_reset = False

    if is_bot:
        if session['bot_strength'] < threat_level:
            reward = 10.0
            print(f"Bot {client_id} caught! Threat: {threat_level}/10. REWARD: {reward}. Resetting session.")
            done = True
            session_reset = True
            session['captchas_solved'] = 0  # Reset the bot's state
            session['last_captcha_solved'] = -1
            del client_sessions[client_id]  # Remove client id from sessions
        else:
            reward = -3.0
            print(f"Bot {client_id} survived. Threat: {threat_level}/10. PUNISHMENT: {reward}")
            session['captchas_solved'] += 1  # Bot survived, increment counter
            session['last_captcha_solved'] = threat_level
    else:
        reward = 5 - threat_level
        print(f"Human user. Threat: {threat_level}/10. Reward: {reward}")
        session['captchas_solved'] += 1
        session['last_captcha_solved'] = threat_level

    return jsonify({
        "status": "processed",
        "threat_level": int(threat_level),
        "humanity_score": humanity_score,
        "session_reset": session_reset,
        "bot_strength": session.get('bot_strength', None)
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
            if COLLECT_DATA:
                json.dump(data, f, indent=4)

        return jsonify({"status": "success", "message": f"Data saved to {filename}"}), 200
    except Exception as e:
        print(f"Error collecting data: {e}")
        return jsonify({"status": "error", "message": "An internal error occurred"}), 500


def handle_interrupt(signal, frame):
    """Handles the KeyboardInterrupt signal gracefully."""
    print("\nCaught KeyboardInterrupt. Exiting...")
    shutdown_hook()  # Manually invoke the shutdown hook when interrupted
    exit(0)

# --- Graceful Shutdown ---
def shutdown_hook():
    """This function is called when the application is shutting down."""
    # if not REWARD_MEMORY:
    #     print("Skipping false shutdown...")
    #     return

    print("Shutting down gracefully...")
    rl_agent.save_model(MODEL_SAVE_PATH)

    # cur_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # reward_metric_filename = f"reward_metric_{cur_time}.json"
    # with open(reward_metric_filename, "w") as f:
    #     json.dump(REWARD_MEMORY, f, indent=4)
    #     print(REWARD_MEMORY)
    # print(f"Reward metrics saved to file {reward_metric_filename}!!!")


if __name__ == '__main__':
    # Load the RL model on startup
    rl_agent.load_model(MODEL_SAVE_PATH)
    # Register the shutdown hook to save the model on exit
    atexit.register(shutdown_hook)

    # Register a signal handler for KeyboardInterrupt
    # signal.signal(signal.SIGINT, handle_interrupt)

    # Run the Flask app
    app.run(debug=True, host="100.84.51.104", port=5000)
