import requests

API_URL = "http://localhost:11434/api/generate"  # Ollama API, not WebUI

def send_message(user_message):
    data = {
        "model": "llama3.2:3b",
        "prompt": user_message,
        "stream": False
    }

    response = requests.post(API_URL, json=data)
    return response.json()["response"]

# Example conversation
response1 = send_message("Hello, how are you?")
print(response1)

response2 = send_message("What did I ask you before?")
print(response2)
