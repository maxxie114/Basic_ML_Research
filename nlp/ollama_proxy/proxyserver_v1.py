from flask import Flask, request, jsonify
import requests
import logging
from logging.config import dictConfig
from flask_cors import CORS  # Allow cross-origin requests (needed for Page Assist)
import flask

# Configure detailed logging
dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'DEBUG',  # Ensure detailed logging
        'handlers': ['wsgi']
    }
})

app = Flask(__name__)
CORS(app)  # Enable CORS to prevent browser restrictions

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


@app.before_request
def log_request_info():
    """Logs incoming request details before processing."""
    app.logger.info('Received %s request to %s', request.method, request.url)
    app.logger.debug('Request Headers: %s', dict(request.headers))
    app.logger.debug('Request Body: %s', request.get_data(as_text=True))

@app.after_request
def log_response_info(response):
    """Logs outgoing response details before sending it back."""
    app.logger.info('Responded with status code %s', response.status)
    app.logger.debug('Response Headers: %s', dict(response.headers))
    app.logger.debug('Response Body: %s', response.get_data(as_text=True))
    return response

@app.errorhandler(Exception)
def handle_exception(e):
    """Handles unexpected errors and logs them."""
    app.logger.error('An unhandled exception occurred: %s', e, exc_info=True)
    return jsonify({'error': 'Internal Server Error', 'message': str(e)}), 500

# Define proxy routes
@app.route('/proxy', methods=['GET'])
def proxy_root():
    """Optional: Handles direct access to /proxy."""
    return jsonify({'message': 'Proxy root endpoint'})

@app.route('/proxy/api/chat', methods=['POST'])
def proxy_chat():
    """Handles chat-based API requests."""
    # auth_response = validate_api_key()
    # if auth_response:
    #    return auth_response
    return forward_request(OLLAMA_CHAT_URL, 'POST', flask.request.json)

@app.route('/proxy/api/generate', methods=['POST'])
def proxy_generate():
    """Handles text generation API requests."""
    auth_response = validate_api_key()
    if auth_response:
        return auth_response
    return forward_request(OLLAMA_CHAT_URL, "POST", flask.request.json)

@app.route('/proxy/api/tags', methods=['GET'])
def proxy_tags():
    """Handles retrieval of available Ollama models."""
    auth_response = validate_api_key()
    if auth_response:
        return auth_response
    return forward_request(OLLAMA_TAGS_URL, 'GET')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=False)  # Run with debug mode for better error tracking
