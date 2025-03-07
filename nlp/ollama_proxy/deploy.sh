#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to display status messages
status() {
    echo -e "\n\033[1;34m$1\033[0m\n"
}

# Function to display error messages
error() {
    echo -e "\n\033[1;31m$1\033[0m\n" >&2
    exit 1
}

# 1. Update and install necessary system packages
status "Updating system packages and installing dependencies..."
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv jq screen curl

# 2. Install Ollama
status "Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
    status "Ollama installed successfully."
else
    status "Ollama is already installed."
fi

# 3. Install deepseek-r1:70b model
status "Installing deepseek-r1:70b model..."
ollama pull deepseek-r1:70b
status "Model deepseek-r1:70b installed successfully."

# 4. Start Ollama server
status "Starting Ollama server..."
if pgrep -x "ollama" > /dev/null; then
    status "Ollama server is already running."
else
    ollama serve &
    status "Ollama server started successfully."
fi

# First copy the config json template file as config.json
cp config_template.json config.json

# 5. Generate a new API key and update config.json
status "Generating a new API key..."
NEW_API_KEY=$(head -c 32 /dev/urandom | sha256sum | awk '{print $1}')
CONFIG_FILE="config.json"

if [[ -f "$CONFIG_FILE" ]]; then
    jq --arg new_key "$NEW_API_KEY" '.api_key = $new_key' "$CONFIG_FILE" > tmp.$$.json && mv tmp.$$.json "$CONFIG_FILE"
    status "config.json updated with new API key."
else
    error "config.json file not found."
fi

# 6. Set up Python virtual environment and install dependencies
status "Setting up Python virtual environment and installing dependencies..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
if [[ -f "requirements.txt" ]]; then
    pip install -r requirements.txt
    status "Python dependencies installed successfully."
else
    error "requirements.txt file not found."
fi
deactivate

# 7. Run proxyserver_v4.py in a screen session
status "Starting proxyserver_v4.py in a screen session..."
SCREEN_NAME="proxy_server"
PROXY_SCRIPT="proxyserver_v4.py"

if screen -list | grep -q "$SCREEN_NAME"; then
    status "Screen session '$SCREEN_NAME' is already running."
else
    if [[ -f "$PROXY_SCRIPT" ]]; then
        screen -dmS "$SCREEN_NAME" bash -c "source venv/bin/activate && python3 $PROXY_SCRIPT"
        status "proxyserver_v4.py is running in screen session '$SCREEN_NAME'."
    else
        error "$PROXY_SCRIPT not found."
    fi
fi

status "Deployment completed successfully."
