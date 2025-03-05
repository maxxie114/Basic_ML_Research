# Implementation of a flask proxy server to proxy the request to ollama server
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# Define your API key
API_KEY = 'c852148fa0f83063009c0b6c46e8bd2c65cfecba02076325c99f043eb6cf912c'

# Ollama API endpoint
OLLAMA_API_URL = 'http://localhost:11434/v1/completions'

@app.route('/proxy', methods=['POST'])
def proxy():
    # Check for API key in headers
    client_api_key = request.headers.get('X-API-Key')
    if client_api_key != API_KEY:
        return jsonify({'error': 'Unauthorized'}), 401

    # Forward the request to the Ollama API
    try:
        response = requests.post(OLLAMA_API_URL, json=request.json)
        return jsonify(response.json()), response.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

