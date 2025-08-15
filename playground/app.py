from flask import Flask, request, jsonify, render_template
import json
import os
import time
from datetime import datetime

def format_timestamp(ms_timestamp):
    dt_local = datetime.fromtimestamp(ms_timestamp / 1000.0)
    formatted_time = dt_local.strftime('%Y-%m-%d_%H-%M-%S')   # Format to string using strftime
    return formatted_time

app = Flask(__name__)

# --- Directory Setup ---
DATA_DIR = 'data'
BLOGS_DIR = 'blogs'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(BLOGS_DIR):
    os.makedirs(BLOGS_DIR)
    # Create a sample blog post if the directory is new
    sample_blog = {
        "id": 1,
        "topic": "Technology",
        "heading": "The Rise of Adaptive Security",
        "author": "Jane Doe",
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


@app.route('/api/collect', methods=['POST'])
def collect_data():
    """Receives user interaction data and saves it to a file."""
    try:
        data = request.get_json()
        print(data)
        if not data or 'timestamps' not in data or 'start' not in data['timestamps']:
            return jsonify({"status": "error", "message": "Invalid data format"}), 400

        if data.get('bot_info', None):
            timestamp = data.get('bot_info', {}).get('session_timestamp', data['timestamps']['start'])
            formatted_timestamp = format_timestamp(timestamp)
            bot_name = data.get('bot_info', {}).get('bot_name', None)
            filename = f"bot_{bot_name}_data_{formatted_timestamp}.json"
        else:
            timestamp = data['timestamps'].get('start', datetime.fromtimestamp(time.time()))
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
    app.run(debug=True, port=5000)
