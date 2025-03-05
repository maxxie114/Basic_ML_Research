# Test deepseek-r1 proxy set up on a remote server
# The proxy use a flask app with api_key authentication to avoid being able to directly reach the ollama server
import requests

# Proxy server URL
PROXY_URL = 'http://107.170.79.167:5000/proxy'

# Your API key
API_KEY = 'c852148fa0f83063009c0b6c46e8bd2c65cfecba02076325c99f043eb6cf912c'

# Sample prompt
prompt = "Explain the concept of quantization in machine learning."

# Request payload
data = {
    'model': 'deepseek-r1:7b',
    'prompt': prompt,
    'temperature': 0.7,
    'max_tokens': 5000
}

# Headers with API key
headers = {
    'Content-Type': 'application/json',
    'X-API-Key': API_KEY
}

# Send request to the proxy server
response = requests.post(PROXY_URL, json=data, headers=headers)

if response.status_code == 200:
    print(response.json()['choices'][0]['text'])
else:
    print(f"Error {response.status_code}: {response.text}")

