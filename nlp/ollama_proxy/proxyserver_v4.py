import flask
import requests
import logging
import sys
from flask import Response, stream_with_context, request, jsonify

# Configure logging to both console and file
log_file = "proxy.log"

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Logs to console
        logging.FileHandler(log_file, mode='a', encoding='utf-8')  # Logs to file
    ]
)

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
        logging.warning('Unauthorized access attempt')
        return jsonify({'error': 'Unauthorized'}), 401
    return None

@app.before_request
def log_request_info():
    """Logs incoming request details before processing."""
    logging.info(f'Received {request.method} request to {request.url}')
    logging.debug(f'Request Headers: {dict(request.headers)}')

    # ✅ FIXED: Use get_json() instead of get_data() to avoid consuming the request body
    try:
        json_body = request.get_json(silent=True)
        if json_body:
            logging.debug(f'Request JSON Body: {json_body}')
    except Exception as e:
        logging.debug(f'Could not parse request body as JSON: {str(e)}')

def forward_request(url, method, data=None):
    """Forwards requests to the Ollama API and ensures true streaming responses."""
    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.request(method, url, json=data, headers=headers, stream=True)

        response.raise_for_status()

        # ✅ Log response metadata BEFORE returning, so no need for @after_request
        logging.info(f'Forwarding request to {url} - Response Status: {response.status_code}')
        logging.debug(f'Response Headers: {response.headers}')

        def generate():
            for line in response.iter_lines():
                if line:
                    try:
                        json_data = line.decode('utf-8')
                        log_message = f"STREAMING: {json_data}"
                        print(log_message, flush=True)  # 🔥 Immediate log output
                        logging.info(log_message)  # 🔥 Log to file
                        yield json_data + "\n"
                    except Exception as e:
                        error_message = f"Error processing stream: {e}"
                        logging.error(error_message)
                        print(error_message, flush=True)

        return Response(stream_with_context(generate()), content_type='application/json')

    except requests.exceptions.RequestException as e:
        logging.error(f"Error forwarding request to {url}: {e}", exc_info=True)
        return jsonify({'error': 'Bad Gateway', 'message': str(e)}), 502

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

