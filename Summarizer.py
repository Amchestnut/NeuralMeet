import time
import requests
import sys
import tiktoken
import json


class Summarizer:
    API_URL = "http://localhost:11434/api/generate"

    # Parameters: can adjust this later
    CHUNK_TOKEN_SIZE = 3000         # Maximum tokens per chunk from the transcript
    CHUNK_SUMMARY_TOKENS = 500      # Expected token length for each chunk's summary
    CONTEXT_SUMMARY_TOKENS = 300    # Expected token length for the rolling context summary
    FINAL_SUMMARY_THRESHOLD = 4000  # Maximum tokens allowed for final summary input
    REDUCTION_CHUNK_SIZE = 2000     # Maximum tokens for each group during history reduction

    def __init__(self, file_type="meeting"):
        """
        file_type can be 'meeting', 'lecture', or 'call'.
        We will adjust our prompts based on this
        """
        self.file_type = file_type.lower().strip()

        if tiktoken:
            try:
                self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
            except Exception:
                self.tokenizer = None
        else:
            self.tokenizer = None

        self.chunk_prompt_templates = {
            "meeting": (
                "System: You are a precise assistant that captures ALL details and returns them in JSON format with exactly two keys: 'chunk_part' and 'updated_context'.\n\n"
                "Meeting Transcript Chunk:\n{chunk_text}\n\n"
                "Current Context Summary:\n{context_summary}\n\n"
                "Instructions:\n"
                "1. PRESERVE AND EXPAND on every single piece of information. Do not summarize or condense.\n"
                "2. Include and elaborate on:\n"
                "   - EVERY decision made (what was decided, why, implications)\n"
                "   - ALL discussion points (even if they seem minor)\n"
                "   - EVERY action item (who, what, when, how)\n"
                "   - ALL numbers, data points, and metrics mentioned\n"
                "   - ANY concerns or issues raised\n"
                "   - EVERY person mentioned and their role/contribution\n"
                "   - ALL deadlines, dates, and time-sensitive information\n"
                "3. Structure the output using Markdown:\n"
                "   - Use '## Topic' for main sections\n"
                "   - Use '### Subtopic' for subsections\n"
                "   - Use '* ' for main bullet points\n"
                "   - Use '  - ' for sub-bullet points\n"
                "   - Use '**bold**' for important points\n"
                "   - Use '*italic*' for emphasis\n"
                "   - Add '\\n' for proper line spacing\n"
                "4. Organize content into clear sections with headers\n"
                "5. Use bullet points for lists of items\n"
                "6. Aim for maximum detail - use all 500 tokens\n"
                "7. Return in this exact JSON format:\n"
                "{{\n"
                "  \"chunk_part\": \"[DETAILED 500-token structured Markdown text preserving all information]\",\n"
                "  \"updated_context\": \"[300-token ongoing narrative of key themes and developments]\"\n"
                "}}\n"
            ),
            "lecture": (
                "System: You are a precise assistant that captures ALL details and returns them in JSON format with exactly two keys: 'chunk_part' and 'updated_context'.\n\n"
                "Lecture Transcript Chunk:\n{chunk_text}\n\n"
                "Current Context Summary:\n{context_summary}\n\n"
                "Instructions:\n"
                "1. PRESERVE AND EXPAND on every piece of information. Do not summarize or condense.\n"
                "2. Include and elaborate on:\n"
                "   - EVERY concept introduced (with full explanation)\n"
                "   - ALL examples given (with complete context)\n"
                "   - EVERY definition provided\n"
                "   - ALL relationships between concepts\n"
                "   - ANY questions asked and answers given\n"
                "   - EVERY analogy or comparison used\n"
                "   - ALL technical terms and their explanations\n"
                "3. Structure the output using Markdown:\n"
                "   - Use '## Topic' for main concepts\n"
                "   - Use '### Subtopic' for related points\n"
                "   - Use '* ' for main examples/points\n"
                "   - Use '  - ' for detailed explanations\n"
                "   - Use '**bold**' for key terms\n"
                "   - Use '*italic*' for definitions\n"
                "   - Add '\\n' for proper line spacing\n"
                "4. Organize content into logical sections\n"
                "5. Use bullet points for examples and explanations\n"
                "6. Aim for maximum detail - use all 500 tokens\n"
                "7. Return in this exact JSON format:\n"
                "{{\n"
                "  \"chunk_part\": \"[DETAILED 500-token structured Markdown text preserving all information]\",\n"
                "  \"updated_context\": \"[300-token ongoing narrative of key concepts and their development]\"\n"
                "}}\n"
            ),
            "call": (
                "System: You are a precise assistant that captures ALL details and returns them in JSON format with exactly two keys: 'chunk_part' and 'updated_context'.\n\n"
                "Phone Call Transcript Chunk:\n{chunk_text}\n\n"
                "Current Context Summary:\n{context_summary}\n\n"
                "Instructions:\n"
                "1. PRESERVE AND EXPAND on every single piece of information. Do not summarize or condense.\n"
                "2. Include and elaborate on:\n"
                "   - EVERY topic discussed (with full context)\n"
                "   - ALL agreements or commitments made\n"
                "   - EVERY request or requirement mentioned\n"
                "   - ALL follow-up items and their details\n"
                "   - ANY concerns or issues raised\n"
                "   - EVERY person mentioned and their role\n"
                "   - ALL numbers, dates, and specific details\n"
                "   - EVERY business term or reference\n"
                "3. Structure the output using Markdown:\n"
                "   - Use '## Call Overview' for the main section\n"
                "   - Use '### Topic' for each major discussion point\n"
                "   - Use '* ' for important statements\n"
                "   - Use '  - ' for details and follow-ups\n"
                "   - Use '**bold**' for commitments/decisions\n"
                "   - Use '*italic*' for important details\n"
                "   - Add '\\n' for proper line spacing\n"
                "4. Organize content chronologically or by topic\n"
                "5. Use bullet points for action items and details\n"
                "6. Aim for maximum detail - use all 500 tokens\n"
                "7. Return output in this exact JSON format dictionary, without any additional text:\n"
                "{{\n"
                "  \"chunk_part\": \"[Write structured Markdown text preserving all important information discussed]\",\n"
                "  \"updated_context\": \"[just some 300-token ongoing narrative of the call's progression]\"\n"
                "}}\n"
            )
        }


        self.final_prompt_templates = {
            "meeting": (
                "You are creating a final, detailed, concise and comprehensive **Meeting** report.\n"
                "Include all important discussions, decisions, tasks, etc.\n\n"
                "Below are all the important information that was discussed:\n\n{combined_history}\n\n"
                "Instructions:\n"
                "1. Write an **extensive** yet well-organized report. Keep as much detail as possible.\n"
                "2. Remove only repeated or trivial filler.\n"
                "3. Include sections or bullet points covering:\n"
                "   - Major decisions, ownership of tasks, and due dates\n"
                "   - Key facts, figures (e.g. financial data, metrics)\n"
                "   - Project progress or updates\n"
                "   - Action items and next steps\n"
                "   - Any unresolved issues or questions\n"
                "   - Any important data related to the money must be included and correct. \n"
                "4. Maintain a clear structure (use headings or bullet points) so it's easy to read.\n"
                "5. Use clear headings and bullet points (Markdown symbols like '*', '**', '#', '##') for readability.\n"
                "6. Do not output JSON—output plain text.\n"
                "7. Ensure that the final text is cohesive and complete.\n"
                "Feel free to write an output even 4000 tokens long.\n"
            ),
            "lecture": (
                "You are creating a final, detailed, concise and comprehensive **Lecture** report.\n"
                "This should capture key topics, explanations, highlight any examples that were given, and explain them if they were important, and insights.\n\n"
                "Below are all the important information that was discussed:\n\n{combined_history}\n\n"
                "Instructions:\n"
                "1. Write a thorough but organized summary of the lecture material.\n"
                "2. Retain important details, focusing on concepts explained, examples given, and any critical definitions.\n"
                "3. Use headings or bullet points to structure the summary (e.g., # Topics, ## and *text* for Examples, use '**' for Key Points).\n"
                "4. Do **not** output JSON—only plain text.\n"
                "   Use symbols like '*', '**', '#', '##', '###' and other Markdown symbols so it's easily used in Obsidian, but focus most on '*', '**' and '#', '##'.\n"
                "5. Keep it cohesive, removing only obvious repetition.\n"
                "6. You must not lose any data, only organize the given data and highlight it better.\n"
                "Feel free to write an output even 4000 tokens long.\n"
            ),
            "call": (
                "You are creating a final, detailed, concise and comprehensive **Phone call** report.\n"
                "It should capture the key points of the conversation, follow-ups, and next steps.\n\n"
                "Below are all the important information that was discussed\n\n{combined_history}\n\n"
                "Instructions:\n"
                "1. Provide a clear, organized recap of the phone call.\n"
                "2. Highlight important details:\n"
                "   - Key discussion items\n"
                "   - Agreements or commitments made\n"
                "   - Action items or due dates\n"
                "3. Do **not** output JSON—only plain text.\n"
                "   Use symbols like '*', '**', '#', '##', '###' and other Markdown symbols so it's easily used in Obsidian, but focus most on '*', '**' and '#', '##'.\n"
                "4. Keep it cohesive, removing only obvious repetition.\n"
                "5. You must not lose any data, only organize the given data and highlight it better.\n"
                "Feel free to write an output even 4000 tokens long.\n"
            )
        }


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
        Sends the chunk and current context summary to the API.
        Expects a JSON response with two keys: "chunk_part" and "updated_context".
        """
        template = self.chunk_prompt_templates.get(self.file_type, self.chunk_prompt_templates["meeting"])
        prompt = template.format(
            chunk_text=chunk_text,
            context_summary=context_summary if context_summary else "No previous context."
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
            response_text = data.get('response', '')

            # Debugging the response
            print("\n=== Raw API Response ===")
            print(f"Response text: {response_text}")

            try:
                response_text = response_text.strip()

                print("\n=== Attempting to parse JSON ===")
                print(f"Cleaned response text: {response_text}")

                parsed_response = json.loads(response_text)

                # Successfully parsed JSON
                print("\n=== Parsed JSON ===")
                print(f"Parsed response: {json.dumps(parsed_response, indent=2)}")

                # Now we expect two keys: "chunk_part" and "updated_context"
                chunk_part = parsed_response.get('chunk_part', '')
                updated_context = parsed_response.get('updated_context', '')

                print("\n=== Generated Summary ===")
                print(f"PROMPT:\n{prompt}")
                print(f"\nRESPONSE:\n{json.dumps(parsed_response, indent=2)}")
                print(f"\nCHUNK_PART:\n{chunk_part}")
                print(f"\nUPDATED_CONTEXT:\n{updated_context}")

                return chunk_part, updated_context

            except json.JSONDecodeError as e:
                print(f"\n=== JSON Parse Error ===")
                print(f"Error details: {str(e)}")
                print(f"Failed to parse text: {response_text}")

                with open("ERROR_OCCURED", "a", encoding="utf-8") as f:
                    f.write(f"Yeah, some error occured: {response_text}\n")

                chunk_part = response_text
                updated_context = context_summary
                return chunk_part, updated_context

        except Exception as e:
            print("Error processing chunk:", e)
            with open("ERROR_OCCURED 2", "a", encoding="utf-8") as f:
                f.write(f"Yeah, some error occured........\n")
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
            # Extract just the response text from llama response format
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
                    # We repeat the process of summarizing from the beginning, but not with the original text, but rather with the already summarized text, so we don't exceed 4000 tokens
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
        Creates the final big, organized report using the combined chunk summaries.
        Focus on retaining as much important detail as possible, removing only duplicates.
        """
        combined_history = " ".join(history_list)

        template = self.final_prompt_templates.get(self.file_type, self.final_prompt_templates["meeting"])
        prompt = template.format(combined_history=combined_history)

        print("We are now printing the PROMPT for the FINAL SUMMARY, it also includes ALL HISTORY SO FAR")
        print(prompt)

        payload = {
            "model": "llama3.2:3b",
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(self.API_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            final_text = data.get('response', '').strip()
            return final_text
        except Exception as e:
            print("Error generating final summary:", e)
            return combined_history


    def summarize(self, full_text):
        """
        Main entry point:
          1. Split the text into chunks.
          2. Process each chunk to get chunk_part, update context_summary.
          3. Optionally reduce if the combined chunk summaries are too large.
          4. Generate one final extensive meeting report with minimal repetition.
        """
        context_summary = ""  # initial context is empty
        history_list = []

        # Split the full transcript into chunk_of_text
        chunks = self._split_text_into_chunks(full_text, self.CHUNK_TOKEN_SIZE)
        print(f"Total chunks to process: {len(chunks)}")

        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i + 1} of {len(chunks)}...")
            chunk_part, updated_context = self._process_chunk(chunk, context_summary)
            print("AJDE DA PRINTAMO CHUNK SUMMARY:", chunk_part)
            history_list.append(chunk_part)
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

    something = "asfas asfsafs asf asf as fasf asfasfasf"
    summarizer = Summarizer()
    res = summarizer._count_tokens(something)
    print(res)
    print(summarizer._split_text_into_chunks(something, summarizer.CHUNK_TOKEN_SIZE))
