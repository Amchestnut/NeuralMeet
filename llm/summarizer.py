import re
import requests
import tiktoken
from prompt_factory import PromptFactory


class Summarizer:
    """
    This class is responsible for:
      1) Interacting with a Large Language Model (LLM).
      2) Summarizing text chunks using specific prompt templates.
      3) Generating a final report from multiple chunk summaries.
    """

    API_URL = "http://localhost:11434/api/generate"

    # Adjustable parameters
    CHUNK_SUMMARY_TOKENS = 500      # Expected token length for each chunk's summary
    CONTEXT_SUMMARY_TOKENS = 300    # Expected token length for the rolling context summary
    FINAL_SUMMARY_THRESHOLD = 4000  # Maximum tokens allowed for final summary input
    REDUCTION_CHUNK_SIZE = 2000     # Maximum tokens for each group during history reduction

    def __init__(self, model_name="llama3.1:latest"):
        # LLM that we will be using
        self.model_name = model_name

        # Attempt to use tiktoken for more accurate token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception:
            self.tokenizer = None

        # We use a PromptFactory where all prompt templates are stored
        self.prompt_factory = PromptFactory()

    def _count_tokens(self, text: str) -> int:
        """
        Count the approximate number of tokens in the text.
        If tiktoken is available, use it; otherwise, fallback to simple word count.
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text.split())

    def _call_llm(self, prompt: str) -> str:
        """
        A generic method for sending a prompt to the LLM endpoint (local or remote)
        and retrieving the response.

        :param prompt: The complete text/prompt to be sent to the model.
        :return: The model's response as a string.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        try:
            resp = requests.post(self.API_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
        except Exception as e:
            print(f"Error while calling LLM: {e}")
            return ""

    def process_chunk(self, chunk_text: str, context_summary: str, file_type: str) -> (str, str):
        """
        Process a transcript chunk, returning two parts:
          1) CHUNK_PART: summarized content from this chunk.
          2) UPDATED_CONTEXT: updated rolling context.

        We use a delimiter-based approach. The prompt instructs the LLM to output:
           CHUNK_PART: ...
           ---DELIMITER---
           UPDATED_CONTEXT: ...
        If that delimiter is not found, we fallback to raw text for CHUNK_PART, and keep the same context.

        :param chunk_text: The actual transcript text for this chunk.
        :param context_summary: The rolling context summary so far.
        :param file_type: 'meeting', 'lecture', or 'phone call'.
        """

        # Get the prompt template for chunk processing
        prompt_template = self.prompt_factory.get_chunk_prompt(file_type)

        # Fill the template with the actual chunk and context
        prompt_filled = prompt_template.format(
            chunk_text=chunk_text,
            context_summary=context_summary
        )

        raw_text = self._call_llm(prompt_filled)
        if not raw_text:
            return "", context_summary

        with open("raw_text.txt", "w", encoding="utf-8") as f:
            f.write(raw_text)

        # Parsing the two sections using a delimiter approach with '---'
        pattern = re.compile(
            r"[\*#\s]*CHUNK_PART[\*#\s]*\s*(.*?)\s*---\s*[\*#\s]*UPDATED_CONTEXT[\*#\s]*\s*(.*)$",
            re.DOTALL
        )

        match = pattern.search(raw_text)
        if match:
            chunk_part = match.group(1).strip()
            updated_context = match.group(2).strip()
            return chunk_part, updated_context

        # If delimiter not found, fallback
        return raw_text, context_summary

    def _summarize_text(self, text: str, target_length: int) -> str:
        """
        Summarize the given text into approximately `target_length` tokens.
        We keep it simple: a direct prompt to the LLM asking for a concise summary.
        """
        prompt = (
            f"Summarize the following text in around {target_length} tokens:\n\n{text}\n\n"
            "Provide only the summary text."
        )
        return self._call_llm(prompt)

    def reduce_history(self, history_list):
        """
        This extra functions will used be only if the user wants a SHORT FINAL REPORT.

        Iteratively reduces the length of the chunk summaries by summarizing them in groups
        until the total token count is below FINAL_SUMMARY_THRESHOLD.
        For example:
            - Final summary has 7000 tokens.
            - Bring the summary down to 4000 tokens.

        :param history_list: A list of chunk-based summaries (strings).
        :return: A reduced version of history_list that fits under the token threshold.
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
                    # Summarize this group down to CONTEXT_SUMMARY_TOKENS
                    reduced_summary = self._summarize_text(group_text, self.CONTEXT_SUMMARY_TOKENS)
                    new_history_list.append(reduced_summary)
                    current_group = [summary]
                    current_tokens = summary_tokens
                else:
                    current_group.append(summary)
                    current_tokens += summary_tokens

            if current_group:
                group_text = " ".join(current_group)
                reduced_summary = self._summarize_text(group_text, self.CONTEXT_SUMMARY_TOKENS)
                new_history_list.append(reduced_summary)

            history_list = new_history_list
            combined_history = " ".join(history_list)

        return history_list

    def final_summary(self, history_list, file_type: str) -> str:
        """
        Creates a final, comprehensive report from the combined chunk summaries in `history_list`.

        :param history_list: List of all chunk_part texts.
        :param file_type: 'meeting', 'lecture', or 'call'.
        :return: The final, big organized report as a string.
        """
        combined_history = " ".join(history_list)
        final_prompt_template = self.prompt_factory.get_final_prompt(file_type)
        prompt = final_prompt_template.format(combined_history=combined_history)

        final_text = self._call_llm(prompt)
        return final_text if final_text else combined_history
