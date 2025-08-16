import requests

# FastAPI server URL
url = "http://127.0.0.1:8000/predict"

# Data to send (must match InputData model)
payload = {
    "mouse_movement": [
        [1, 2],
        [3, 4],
        [5, 6]
    ]
}

# Make POST request
response = requests.post(url, json=payload)

# Print response
if response.status_code == 200:
    print("Prediction:", response.json()["prediction"])
else:
    print("Error:", response.status_code, response.text)
