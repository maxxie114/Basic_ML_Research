import requests

url = "http://localhost:11434/api/chat"
headers = {"Content-Type": "application/json"}
data = {
    "model": "deepseek-r1:70b",
    "options": {},
    "messages": [{"role": "user", "content": "hi"}]
}

response = requests.post(url, json=data, headers=headers, stream=True)

print("Waiting for response...")
for line in response.iter_lines(decode_unicode=True):
    if line:
        print("Received chunk:", line)  # Debugging output

