import time
import requests

# --- Configuration ---
API_URL = "http://localhost:11434/api/generate"  # Summarization API endpoint (your AI service)
TOKEN_LIMIT = 4000  # Maximum tokens per prompt (example value)
CHUNK_TIME_SECONDS = 60  # Summarize every 60 seconds (adjust as needed)
CHUNK_TOKEN_THRESHOLD = 1000  # Alternatively, summarize if current chunk exceeds this token count

# --- Global State ---
meeting_summaries = []  # To store summary segments
current_chunk = ""  # Current transcript text chunk
chunk_start_time = time.time()


# --- Helper Functions ---
def estimate_tokens(text):
    """Estimate token count (approx. 1 token per 4 characters)."""
    return len(text) // 4


def summarize_text(text):
    """
    Call the summarization API to summarize the provided text.
    Adjust the prompt as needed to highlight key meeting points and client requirements.
    """
    summary_prompt = f"Summarize the following meeting segment, highlighting key points, decisions, and action items:\n\n{text}\n\nSummary:"
    data = {
        "model": "llama3.2:3b",
        "prompt": summary_prompt,
        "stream": False
    }
    try:
        response = requests.post(API_URL, json=data)
        response_json = response.json()
        summary = response_json.get("response", "").strip()
        return summary if summary else "[No summary returned]"
    except Exception as e:
        print("Error during summarization:", e)
        return "[Error in summarization]"


def process_chunk():
    """Summarize the current chunk and update the global meeting summaries."""
    global current_chunk, chunk_start_time, meeting_summaries
    if current_chunk.strip():
        print("Summarizing current chunk...")
        summary = summarize_text(current_chunk)
        print("Chunk summary:", summary)
        meeting_summaries.append(summary)
        # Reset current chunk and timer
        current_chunk = ""
        chunk_start_time = time.time()


def update_chunk(new_text):
    """Append new transcript text to the current chunk and trigger summarization if needed."""
    global current_chunk, chunk_start_time
    current_chunk += new_text + "\n"
    # Check if it's time to summarize based on time interval or token count
    if (time.time() - chunk_start_time) >= CHUNK_TIME_SECONDS or estimate_tokens(current_chunk) > CHUNK_TOKEN_THRESHOLD:
        process_chunk()


def get_new_transcript():
    """
    Simulate receiving new transcript text from Whisper.
    Replace this simulation with your actual streaming integration.
    """
    # For simulation, returning a fixed snippet.
    return "This is a simulated transcript text from the meeting. Key point: Discuss project roadmap."


# --- Main Processing Loop ---
def main():
    # Simulate a 5-minute meeting (adjust as needed)
    meeting_duration_seconds = 5 * 60
    meeting_end_time = time.time() + meeting_duration_seconds

    while time.time() < meeting_end_time:
        # Simulate receiving transcript text every 5 seconds.
        new_text = get_new_transcript()
        update_chunk(new_text)
        time.sleep(5)  # In production, this would be event-driven based on real audio input

    # Process any remaining transcript text after the meeting ends.
    if current_chunk.strip():
        process_chunk()

    # Generate a final comprehensive meeting summary by combining all summary segments.
    final_summary_text = "\n".join(meeting_summaries)
    print("\nGenerating final meeting summary...")
    final_summary = summarize_text(final_summary_text)
    print("Final Meeting Summary:")
    print(final_summary)


if __name__ == "__main__":
    main()
