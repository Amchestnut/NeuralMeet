import time
import requests


class SummarizerX:
    API_URL = "http://localhost:11434/api/generate"     # AI API endpoint
    TOKEN_LIMIT = 4000                                  # Max tokens per request (adjust as needed)
    CHUNK_TOKEN_THRESHOLD = 2000                        # Summarize when this threshold is reached
    CHUNK_TIME_SECONDS = 60                             # Summarize every 60 seconds

    def __init__(self):
        self.meeting_summaries = []
        self.current_chunk = ""
        self.chunk_start_time = time.time()


    def estimate_tokens(self, text):
        """Estimate token count (approx. 1 token per 4 characters)."""
        return len(text) // 4


    def summarize_text(self, text):
        """Send transcript chunk to AI for summarization."""
        summary_prompt = f"Summarize the following meeting segment, highlighting key points, decisions, and action items:\n\n{text}\n\nSummary:"
        data = {
            "model": "llama3.2:3b",     # My model, but can be changed later to something better
            "prompt": summary_prompt,
            "stream": False
        }
        try:
            response = requests.post(self.API_URL, json=data)
            response_json = response.json()
            return response_json.get("response", "").strip() or "[No summary returned]"
        except Exception as e:
            print("Error during summarization:", e)
            return "[Error in summarization]"


    def process_chunk(self):
        """Summarize the current chunk and store the summary."""
        if self.current_chunk.strip():
            print("Summarizing current chunk...")
            summary = self.summarize_text(self.current_chunk)
            self.meeting_summaries.append(summary)

            print("Chunk Summary:\n", summary)
            self.current_chunk = ""
            self.chunk_start_time = time.time()     # reset timer


    def update_chunk(self, new_text):
        """Append new transcript text to the current chunk and trigger summarization if needed."""
        print("Updating chunk...")
        self.current_chunk += new_text + "\n"
        if (time.time() - self.chunk_start_time) >= self.CHUNK_TIME_SECONDS or self.estimate_tokens(self.current_chunk) > self.CHUNK_TOKEN_THRESHOLD:
            self.process_chunk()


    def split_and_summarize(self, transcript):
        """Break full transcript into chunks dynamically."""
        lines = transcript.split("\n")
        i = 0
        for line in lines:
            i += 1
            print(i)
            self.update_chunk(line)

        # Process any remaining text after all lines are processed
        if self.current_chunk.strip():
            self.process_chunk()


    def get_final_summary(self):
        """Generate a final summary by merging all chunks."""
        print()
        print("Generating final meeting summary...")
        final_summary_text = "\n".join(self.meeting_summaries)
        return self.summarize_text(final_summary_text)
