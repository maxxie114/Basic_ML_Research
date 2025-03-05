from flask import Flask, Response, stream_with_context
import time

app = Flask(__name__)

@app.route('/test-stream')
def test_stream():
    """A simple streaming response to test Flask's ability to stream output."""
    def generate():
        for i in range(10):
            yield f"Chunk {i}\n"
            time.sleep(1)  # Simulate streaming delay

    return Response(stream_with_context(generate()), content_type="text/plain")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=False)

