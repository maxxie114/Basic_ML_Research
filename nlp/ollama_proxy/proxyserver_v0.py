from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# Define your API key
API_KEY = 'c852148fa0f83063009c0b6c46e8bd2c65cfecba02076325c99f043eb6cf912c'

# Ollama API endpoints
OLLAMA_COMPLETIONS_URL = 'http://localhost:11434/v1/completions'
OLLAMA_TAGS_URL = 'http://localhost:11434/api/tags'

def forward_request(url, method, data=None):
    """Helper function to forward requests to the Ollama API."""
    try:
        if method == 'POST':
            response = requests.post(url, json=data)
        else:  # 'GET'
            response = requests.get(url)
        response.raise_for_status()
        return jsonify(response.json()), response.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({'error': str(e)}), 500

@app.route('/proxy', methods=['POST', 'GET'])
def proxy():
    # Check for API key in headers
    client_api_key = request.headers.get('X-API-Key')
    if client_api_key != API_KEY:
        return jsonify({'error': 'Unauthorized'}), 401

    if request.method == 'POST':
        # Forward POST request to the Ollama completions endpoint
        return forward_request(OLLAMA_COMPLETIONS_URL, 'POST', request.json)
    else:  # 'GET'
        # Handle GET requests or return a method not allowed error
        return jsonify({'error': 'GET method not supported on /proxy'}), 405

@app.route('/proxy/api/tags', methods=['GET'])
def proxy_tags():
    # Check for API key in headers
    client_api_key = request.headers.get('X-API-Key')
    if client_api_key != API_KEY:
        return jsonify({'error': 'Unauthorized'}), 401

    # Forward GET request to the Ollama tags endpoint
    return forward_request(OLLAMA_TAGS_URL, 'GET')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

