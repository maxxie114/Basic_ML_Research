import requests
import json
import sys

# Proxy API URL
url = "http://107.170.79.167:5000/proxy/api/chat"

# Headers
headers = {
    "X-API-Key": "c852148fa0f83063009c0b6c46e8bd2c65cfecba02076325c99f043eb6cf912c",
    "Content-Type": "application/json"
}

# Request Data
data = {
    "model": "deepseek-r1:70b",
    "options": {},
    "messages": [{"role": "user", "content": "please guide me step by step on how to do leetcode 126", "images": []}]
}

# Send POST request with streaming enabled
response = requests.post(url, json=data, headers=headers, stream=True)

# Ensure the request was successful
if response.status_code != 200:
    print(f"❌ Error: {response.status_code} - {response.text}")
    sys.exit(1)

print("\n🔵 Streaming Response:\n", end="", flush=True)

# Process the streaming response
try:
    for line in response.iter_lines(decode_unicode=True):
        if line:
            try:
                # Decode and parse JSON line
                json_data = json.loads(line.strip())

                # Extract and print the actual message content if available
                if "message" in json_data and "content" in json_data["message"]:
                    print(json_data["message"]["content"], end="", flush=True)

            except json.JSONDecodeError:
                print("\n⚠️ Warning: Received malformed JSON data.", flush=True)

except KeyboardInterrupt:
    print("\n❌ Stream interrupted by user.")

