import time
import requests
import sys
import tiktoken


class Summarizer:
    API_URL = "http://localhost:11434/api/generate"

    # Parameters: can adjust this later
    CHUNK_TOKEN_SIZE = 2000         # Maximum tokens per chunk from the transcript
    CHUNK_SUMMARY_TOKENS = 500      # Expected token length for each chunk's summary
    CONTEXT_SUMMARY_TOKENS = 300    # Expected token length for the rolling context summary
    FINAL_SUMMARY_THRESHOLD = 4000  # Maximum tokens allowed for final summary input
    REDUCTION_CHUNK_SIZE = 2000     # Maximum tokens for each group during history reduction

    def __init__(self):
        if tiktoken:
            try:
                # Some encoding suitable for my model, example: gpt 3.5 turbo
                self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
            except Exception:
                self.tokenizer = None
        else:
            self.tokenizer = None


    def _count_tokens(self, text):
        """Count tokens using tiktoken if available; otherwise fall back to word count."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            return len(text.split())


    def _split_text_into_chunks(self, text, max_tokens):
        """
        Split the whole text into chunks.
        Using binary search for better time complexity, from O(n^2) to O(n*log n).
        When we find a chunk of the right size, add it.
        """
        chunks = []
        while text:
            # If remaining text is small enough, just add it as the last chunk
            if self._count_tokens(text) <= max_tokens:
                chunks.append(text)
                break

            # Binary search for the split point
            left, right = 0, len(text)
            while left < right - 1:
                mid = (left + right) // 2
                if self._count_tokens(text[:mid]) <= max_tokens:
                    left = mid
                else:
                    right = mid

            # Add the chunk and continue with remaining text
            chunks.append(text[:left])      # add first chunk of size = CHUNK_TOKEN_SIZE, then the second, etc.
            text = text[left:].lstrip()     # lstrip() deletes the spaces at the beginning.

        return chunks


    def _process_chunk(self, chunk_text, context_summary):
        """
        Sends the chunk and the current context summary to the API.
        """
        prompt = (
            f"Meeting Transcript Chunk:\n\n{chunk_text}\n\n"
            f"Current Context Summary (approx {self.CONTEXT_SUMMARY_TOKENS} tokens):\n{context_summary}\n\n"
            "Instructions:\n"
            "1. Provide a concise summary of the above chunk (around 300 tokens).\n"
            "2. Update the context summary to reflect key points from this chunk (around 200 tokens).\n"
            "Return the result in JSON format with keys 'chunk_summary' and 'updated_context'."
        )
        payload = {
            "model": "llama3.2:3b",
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(self.API_URL, json=payload)
            response.raise_for_status()
            data = response.json()

            # Extract the actual response text from Ollama's response format
            response_text = data.get('response', '')

            # Try to parse the response as JSON
            try:
                import json
                parsed_response = json.loads(response_text)
                chunk_summary = parsed_response.get('chunk_summary', '')
                updated_context = parsed_response.get('updated_context', '')
            except json.JSONDecodeError:
                # If parsing fails, use the response as chunk summary
                chunk_summary = response_text
                updated_context = context_summary

            return chunk_summary, updated_context
        except Exception as e:
            print("Error processing chunk:", e)
            return "", context_summary


    def _summarize_text(self, text, target_length):
        """
        Calls the API to summarize the given text to approximately target_length tokens.
        """
        prompt = (
            f"Summarize the following text in a concise manner using around {target_length} tokens:\n\n{text}\n\n"
            "Provide only the summary text."
        )
        payload = {
            "model": "llama3.2:3b",
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(self.API_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            # Extract just the response text from Ollama's response format
            summary = data.get('response', '').strip()
            return summary
        except Exception as e:
            print("Error summarizing text:", e)
            return text


    def _reduce_history(self, history_list):
        """
        Iteratively reduce the list of history summaries until the combined token count is below the FINAL_SUMMARY_THRESHOLD.
        Groups summaries into chunks and summarizes each group.
        """
        combined_history = " ".join(history_list)
        while self._count_tokens(combined_history) > self.FINAL_SUMMARY_THRESHOLD:
            new_history_list = []
            current_group = []
            current_tokens = 0

            for summary in history_list:
                summary_tokens = self._count_tokens(summary)
                if current_tokens + summary_tokens > self.REDUCTION_CHUNK_SIZE:
                    group_text = " ".join(current_group)
                    reduced_summary = self._summarize_text(group_text, target_length=self.CONTEXT_SUMMARY_TOKENS)
                    new_history_list.append(reduced_summary)
                    current_group = [summary]
                    current_tokens = summary_tokens
                else:
                    current_group.append(summary)
                    current_tokens += summary_tokens

            if current_group:
                group_text = " ".join(current_group)
                reduced_summary = self._summarize_text(group_text, target_length=self.CONTEXT_SUMMARY_TOKENS)
                new_history_list.append(reduced_summary)

            history_list = new_history_list
            combined_history = " ".join(history_list)
        return history_list


    def _final_summary(self, history_list):
        """
        Generates the final comprehensive meeting summary using the combined history list.
        """
        combined_history = " ".join(history_list)
        prompt = (
            f"Based on the following summaries of meeting segments:\n\n{combined_history}\n\n"
            "Generate a final, comprehensive meeting summary that covers all key points."
        )
        payload = {
            "model": "llama3.2:3b",
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(self.API_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            # Extract just the response text from Ollama's response format
            return data.get('response', '').strip()
        except Exception as e:
            print("Error generating final summary:", e)
            return combined_history


    def summarize(self, full_text):
        """
        Main entry point for summarization.
        - Splits the full text into chunks of roughly CHUNK_TOKEN_SIZE tokens.
        - For each chunk, obtains a chunk summary and an updated context summary.
        - Stores each chunk's summary in a history_list.
        - After processing, reduces history if necessary and generates a final summary.
        """
        context_summary = ""  # initial context is empty
        history_list = []

        # Split the full transcript into chunk_of_text
        chunks = self._split_text_into_chunks(full_text, self.CHUNK_TOKEN_SIZE)
        print(f"Total chunks to process: {len(chunks)}")

        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i + 1} of {len(chunks)}...")
            chunk_summary, updated_context = self._process_chunk(chunk, context_summary)
            history_list.append(chunk_summary)
            context_summary = updated_context
            time.sleep(0.5)  # pause to avoid rate limits if necessary

        combined_history_tokens = self._count_tokens(" ".join(history_list))
        print(f"Combined history token count: {combined_history_tokens}")

        # If history is too large, reduce it iteratively.
        if combined_history_tokens > self.FINAL_SUMMARY_THRESHOLD:
            print("Reducing history summaries to meet token limits...")
            history_list = self._reduce_history(history_list)

        # Generate the final meeting summary.
        final_summary = self._final_summary(history_list)
        return final_summary


if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Usage: python summarizer.py <meeting_file.txt>")
    #     sys.exit(1)
    #
    # meeting_file = sys.argv[1]
    # with open(meeting_file, "r", encoding="utf-8") as f:
    #     meeting_text = f.read()
    #
    # summarizer = Summarizer()
    # final_summary = summarizer.summarize(meeting_text)
    # print("\nFinal Meeting Summary:\n")
    # print(final_summary)

    something = "sadasdasdasdaf  asdadfqefwegwsd gwegweg wegweg qqqqqqqqfg gwegweg Count tokens using tiktoken if available; otherwise fall back to word count. Count tokens using tiktoken if available; otherwise fall back to word count. Count tokens using tiktoken if available; otherwise fall back to word count. i dont asd asd asd w w sqqqqw wrt qw www ggg wwww wq wqr ffffasfasfa saf asfas sf w ww wwwwww Count tokens using tiktoken if available; otherwise fall back to word count. i dont asd asd asd w w sqqqqw wrt qw www ggg wwww wq wqr ffffasfasfa saf asfas sf w ww wwwwww Count tokens using tiktoken if available; otherwise fall back to word count. i dont asd asd asd w w sqqqqw wrt qw www ggg wwww wq wqr ffffasfasfa saf asfas sf w ww wwwwww Count tokens using tiktoken if available; otherwise fall back to word count. i dont asd asd asd w w sqqqqw wrt qw www ggg wwww wq wqr ffffasfasfa saf asfas sf w ww wwwwww Count tokens using tiktoken if available; otherwise fall back to word count. i dont asd asd asd w w sqqqqw wrt qw www ggg wwww wq wqr ffffasfasfa saf asfas sf w ww wwwwww Count tokens using tiktoken if available; otherwise fall back to word count. i dont asd asd asd w w sqqqqw wrt qw www ggg wwww wq wqr ffffasfasfa saf asfas sf w ww wwwwww Count tokens using tiktoken if available; otherwise fall back to word count. i dont asd asd asd w w sqqqqw wrt qw www ggg wwww wq wqr ffffasfasfa saf asfas sf w ww wwwwww"
    summarizer = Summarizer()
    res = summarizer._count_tokens(something)
    print(res)
    print(summarizer._split_text_into_chunks(something, summarizer.CHUNK_TOKEN_SIZE))
