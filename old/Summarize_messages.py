import requests

API_URL = "http://localhost:11434/api/generate"  # Ollama API
TOKEN_LIMIT = 4000  # Approximate LLaMA token limit

chat_history = [
    {"role": "system", "content": "You are a helpful AI assistant. Always remember past messages and summarize older ones when needed."}
]

def estimate_tokens(text):
    """Estimate token count (approx 1 token per 4 characters)."""
    return len(text) // 4

def summarize_history():
    """Summarize older messages if the chat history exceeds the token limit."""
    global chat_history

    # Take first 5 messages and summarize them
    first_messages = "\n".join(msg["content"] for msg in chat_history[:5])
    summary_prompt = f"Summarize this conversation so far briefly:\n{first_messages}"

    data = {
        "model": "llama3.2:3b",
        "prompt": summary_prompt,
        "stream": False
    }

    response = requests.post(API_URL, json=data).json()
    summary = response["response"].strip()

    # Replace old messages with the summary
    chat_history[:5] = [{"role": "system", "content": f"Summary: {summary}"}]

def send_message(user_message):
    global chat_history

    chat_history.append({"role": "user", "content": user_message})

    # Format chat history for prompt
    formatted_chat = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
    ) + "\nAssistant:"

    # If token count is too high, summarize old messages
    if estimate_tokens(formatted_chat) > TOKEN_LIMIT:
        summarize_history()

    data = {
        "model": "llama3.2:3b",
        "prompt": formatted_chat,
        "stream": False
    }

    response = requests.post(API_URL, json=data).json()
    bot_reply = response["response"].strip()

    chat_history.append({"role": "assistant", "content": bot_reply})

    return bot_reply

# Example conversation
response1 = send_message("Hello, how are you?")
print("AI:", response1)

response2 = send_message("What did I ask you before?")
print("AI:", response2)

response3 = send_message("Summarize our conversation so far.")
print("AI:", response3)
