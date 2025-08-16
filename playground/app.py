from flask import Flask, request, jsonify, render_template
import json
import os
import time
from datetime import datetime
import random
import requests
# Import the RL agent from our new service
from rl_service import rl_agent

HUMANITY_SCORE_PREDICTOR_URL = "http://127.0.0.1:8000/predict"

def get_humanity_score(mouse_movement):
    if not mouse_movement: return None
    mouse_movement = [[movement['x'], movement['y']] for movement in mouse_movement]
    payload = {"mouse_movement": mouse_movement}
    # Make POST request
    response = requests.post(HUMANITY_SCORE_PREDICTOR_URL, json=payload)

    # Print response
    if response.status_code == 200:
        print("Prediction:", response.json()["prediction"])
        return response.json()["prediction"]
    else:
        print("Error:", response.status_code, response.text)
        return None

def format_timestamp(ms_timestamp):
    dt_local = datetime.fromtimestamp(ms_timestamp / 1000.0)
    formatted_time = dt_local.strftime('%Y-%m-%d_%H-%M-%S')
    return formatted_time

app = Flask(__name__)

# --- In-Memory Session Storage ---
# In a production system, this would be a database like Redis.
client_sessions = {}

# --- Directory Setup ---
DATA_DIR = 'data'
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
    data = request.get_json()
    bot_info = data.get('bot_info')
    is_bot = bot_info is not None

    # Use bot_id for bots, or session start time for humans as a unique identifier
    client_id = bot_info['bot_id'] if is_bot else data['timestamps']['start']

    # Retrieve or initialize the client's session
    if client_id not in client_sessions:
        client_sessions[client_id] = {'captchas_solved': 0}

    session = client_sessions[client_id]

    # --- 1. Humanity Score Calculation (Placeholder) ---
    # mouse_activity = len(data.get('mouse_movements', []))
    # humanity_score = min(1.0, round(mouse_activity / 200.0, 1))
    humanity_score = get_humanity_score(mouse_movement=data.get('mouse_movements', None))
    print("Humaity Score: ", humanity_score)

    # --- 2. State Representation ---
    state = [humanity_score, min(10, session['captchas_solved'])]

    # --- 3. RL Agent Action ---
    threat_level = rl_agent.select_action(state)

    # --- 4. Reward Calculation & Simulation ---
    reward = 0
    done = False
    session_reset = False

    if is_bot:
        if random.random() < (threat_level / 10.0):
            print(f"Bot {client_id} caught! Threat: {threat_level}/10. REWARD +1. Resetting session.")
            reward = 1.0
            done = True
            session_reset = True
            session['captchas_solved'] = 0  # Reset the bot's state
        else:
            print(f"Bot {client_id} survived. Threat: {threat_level}/10. PUNISHMENT -1")
            reward = -1.0
            session['captchas_solved'] += 1  # Bot survived, increment counter
    else:
        reward = 5 - threat_level
        print(f"Human user. Threat: {threat_level}/10. Reward: {reward}")
        session['captchas_solved'] += 1

    # --- 5. Train the RL Agent ---
    next_state = [humanity_score, min(10, session['captchas_solved'])]
    rl_agent.remember(state, threat_level, reward, next_state, done)
    rl_agent.train_model()

    return jsonify({
        "status": "processed",
        "threat_level": threat_level,
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


if __name__ == '__main__':
    app.run(debug=True, host="100.94.176.110", port=5000)
