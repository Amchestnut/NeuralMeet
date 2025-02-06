import requests

API_URL = "http://localhost:11434/api/generate"  # Ollama API

# Chat history (stored manually)
chat_history = [
    {"role": "system", "content": "You are a helpful AI assistant. Remember what the user says."}
]

def send_message(user_message):
    global chat_history

    # Add user message to history
    chat_history.append({"role": "user", "content": user_message})

    # Format chat history properly
    formatted_chat = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
    )

    data = {
        "model": "llama3.2:3b",
        "prompt": formatted_chat,  # Now AI sees full chat history
        "stream": False
    }

    response = requests.post(API_URL, json=data).json()
    bot_reply = response["response"]

    # Add bot response to history
    chat_history.append({"role": "assistant", "content": bot_reply})

    return bot_reply

# Example conversation
response1 = send_message("Hello, how are you?")
print(response1)

response2 = send_message("What did I ask you before?")
print(response2)  # Now, it should remember previous messages!
