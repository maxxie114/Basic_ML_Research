import flask
import requests
import sys
from flask import Response, stream_with_context
from flask import Flask, request, jsonify

app = flask.Flask(__name__)

# API Key for security
API_KEY = 'c852148fa0f83063009c0b6c46e8bd2c65cfecba02076325c99f043eb6cf912c'

# Ollama API endpoints
OLLAMA_BASE_URL = 'http://localhost:11434'
OLLAMA_CHAT_URL = f'{OLLAMA_BASE_URL}/api/chat'
OLLAMA_GENERATE_URL = f'{OLLAMA_BASE_URL}/v1/generate'
OLLAMA_TAGS_URL = f'{OLLAMA_BASE_URL}/api/tags'

def validate_api_key():
    """Validate API Key from request headers."""
    client_api_key = request.headers.get('X-API-Key')
    if client_api_key != API_KEY:
        app.logger.warning('Unauthorized access attempt with API key: %s', client_api_key)
        return jsonify({'error': 'Unauthorized'}), 401
    return None

def forward_request(url, method, data=None):
    """Forwards requests to the Ollama API and handles streaming responses correctly."""
    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.request(method, url, json=data, headers=headers, stream=True)

        # Ensure request was successful
        response.raise_for_status()

        # Stream response handling
        def generate():
            for line in response.iter_lines():
                if line:
                    try:
                        json_data = line.decode('utf-8')  # Decode each line
                        yield json_data + "\n"  # Send each JSON chunk
                    except Exception as e:
                        app.logger.error(f"Error processing stream: {e}")

        return flask.Response(generate(), content_type='application/json')

    except requests.exceptions.RequestException as e:
        app.logger.error('Error forwarding request to %s: %s', url, e, exc_info=True)
        return flask.jsonify({'error': 'Bad Gateway', 'message': str(e)}), 502

@app.route('/proxy/api/chat', methods=['POST'])
def proxy_chat():
    """Handles chat-based API requests."""
    auth_response = validate_api_key()
    if auth_response:
        return auth_response
    return forward_request(OLLAMA_CHAT_URL, 'POST', flask.request.json)

@app.route('/proxy/api/tags', methods=['GET'])
def proxy_tags():
    """Handles retrieval of available Ollama models."""
    auth_response = validate_api_key()
    if auth_response:
        return auth_response
    return forward_request(OLLAMA_TAGS_URL, 'GET')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=False)

