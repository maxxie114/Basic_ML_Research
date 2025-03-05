### **README: Building a Streaming Proxy for Ollama with API Key Authentication**

#### **Overview**
This guide explains how to build a **Flask-based proxy** for **Ollama** to handle **API requests securely** while supporting **real-time streaming responses**. It also documents a critical issue with Flask logging that can **block streaming**, along with the **solution**.

---

## **🚀 Features**
- **Proxy for Ollama's API** (`/api/chat`, `/v1/generate`, `/api/tags`)
- **API Key Authentication** for security
- **Real-time Streaming Support**
- **Detailed Request Logging**
- **Prevents Flask's Logging from Blocking Streaming**

---

## **🔧 How to Build the Proxy**

### **1️⃣ Install Dependencies**
Ensure you have Flask and Requests installed:
```sh
pip install flask requests
```

### **2️⃣ Create the Proxy Server**
Use the following **Flask application** to forward requests to Ollama **without blocking streaming**:

```python
import flask
import requests
import logging
import sys
from flask import Response, stream_with_context, request, jsonify

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s')

app = flask.Flask(__name__)

# API Key for security
API_KEY = 'your_secure_api_key_here'

# Ollama API endpoints
OLLAMA_BASE_URL = 'http://localhost:11434'
OLLAMA_CHAT_URL = f'{OLLAMA_BASE_URL}/api/chat'

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
                        print(f"STREAMING: {json_data}", flush=True)  # 🔥 Immediate log output
                        yield json_data + "\n"
                    except Exception as e:
                        logging.error(f"Error processing stream: {e}")

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
    return forward_request(OLLAMA_CHAT_URL, 'POST', request.json)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=False)
```

---

## **🐛 The Issue We Solved: Flask Logging Blocks Streaming**
During development, we encountered a **critical issue** where **adding response logging (`@after_request`) broke streaming**.

### **❌ The Problem**
- **Flask’s `@after_request` modifies the response**, which causes it to buffer **instead of stream**.
- **Logging the response body forces Flask to read it all before sending it**, preventing real-time responses.

### **✅ The Solution**
- **Moved response logging into `forward_request()` before returning the stream.**
- **Removed `@after_request` to prevent Flask from modifying the streaming response.**
- **Kept request logging (`@before_request`) since it doesn't interfere with streaming.**

---

## **🚀 How to Run the Proxy**
1. **Start the Flask server:**
   ```sh
   python3 your_script.py
   ```
2. **Test Streaming with `curl`**
   ```sh
   curl -X POST http://your-flask-server:5000/proxy/api/chat \
        -H "Content-Type: application/json" \
        -H "X-API-Key: your_api_key" \
        -d '{"model":"deepseek-r1:70b","options":{},"messages":[{"role":"user","content":"hi"}]}' \
        --no-buffer
   ```
3. **Expected Behavior:**
   ✅ Logs requests and responses  
   ✅ Supports streaming responses **without buffering issues**  
   ✅ Secure with API Key authentication  

---

## **📌 Additional Notes**
- **Production Deployment:** Use Gunicorn for better performance:
  ```sh
  gunicorn -w 1 -b 0.0.0.0:5000 --threads 1 --timeout 0 your_script:app
  ```
- **Debugging:** Add `print(..., flush=True)` inside `generate()` to verify streaming chunks.

---

## **🎯 Summary**
| **Feature** | **Status** |
|------------|-----------|
| Secure Proxy for Ollama | ✅ Done |
| API Key Authentication | ✅ Done |
| Real-Time Streaming | ✅ Done |
| Request Logging (`@before_request`) | ✅ Works |
| Response Logging (`@after_request`) | ❌ **Breaks streaming** |
| **🚀 Fix: Move response logging inside `forward_request()`** | ✅ Works! |

This solution provides **real-time streaming** while **keeping API requests secure and logged**. 🎯

---

🚀 **This README documents the exact issue we solved and provides a production-ready proxy for Ollama!** 🚀

---

**Main files included in the Project**
- proxyserver_v0.py: initial version of the proxy server
- proxyserver_v1.py: complete version of the proxy server, recognized by Page Assist as an ollama server, but no streaming
- proxyserver_v2.py: second version of the proxy server, but streaming failed due to the logger
- proxyserver_v3.py: fully functional proxy server, with streaming, and logging
- proxyserver_v4.py: fully functional proxy server, with streaming, logging, and file logging