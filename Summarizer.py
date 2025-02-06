import time
import requests
import sys
import tiktoken


class Summarizer:
    API_URL = "http://localhost:11434/api/generate"

    # Parameters: can adjust this later
    CHUNK_TOKEN_SIZE = 3000         # Maximum tokens per chunk from the transcript
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
            f"Current Context Summary:\n{context_summary}\n\n"
            "Instructions:\n"
            "1. Provide a concise, high-level summary of the above chunk (around 500 tokens).\n"
            "2. Identify and list all key points, focusing on:\n"
            "   - Decisions made\n"
            "   - Future plans or next steps\n"
            "   - Action items (with owners, deadlines, etc.)\n"
            "   - Important data (numbers, performance metrics, etc.)\n"
            "3. Update the context summary with these newly discovered points (around 300 tokens).\n"
            "4. Return the result as JSON with the keys:\n"
            "   {\n"
            "     \"chunk_summary\": \"...\",\n"
            "     \"updated_context\": \"...\"\n"
            "   }"
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

        prompt = (
            "You are creating a final comprehensive meeting report. "
            "It should be detailed, capturing essentially everything important that was discussed, "
            "while omitting repetitive or irrelevant text.\n\n"
            "Below are the chunk summaries:\n\n"
            f"{combined_history}\n\n"
            "Instructions:\n"
            "1. Write an **extensive** yet well-organized report. Keep as much detail as possible.\n"
            "2. Remove only repeated/duplicated statements or trivial filler.\n"
            "3. Include sections or bullet points covering:\n"
            "   - Major decisions, ownership of tasks, and due dates\n"
            "   - Key facts, figures (e.g. financial data, metrics)\n"
            "   - Project progress or updates\n"
            "   - Action items and next steps\n"
            "   - Any unresolved issues or questions\n"
            "4. Maintain a clear structure (use headings or bullet points) so it's easy to read.\n"
            "5. Do **not** output JSON—only plain text that someone could read in a notes tool, "
            "   Use symbols like '*', '**', '#', '##', '###' and other Markdown symbols so it's easily used in Obsidian.\n"
            "6. Make sure the final text is cohesive and does not repeat information unnecessarily."
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
            final_text = data.get('response', '').strip()
            return final_text
        except Exception as e:
            print("Error generating final summary:", e)
            return combined_history


    def summarize(self, full_text):
        """
        Main entry point:
          1. Split the text into chunks.
          2. Process each chunk to get chunk_summary, update context_summary.
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
    lecture = " And we'll back it. I'm I'm online. I'm I'm I'm I'm I'm I'm I'm I'm I'm I'm I'm Can See I can I'm Let's We need to腿 We But we were able to do more, we were able to do more, but we were able to do more, so we decided to do more for the rest of the game. Good, we could shoot this one. I don't know, really, after the first game. I don't really know, I don't really know the rest of the game. I just think we were able to do more for the rest of the game. We could create everything we wanted for the rest of the game. Okay, yes, about So, I don't think that's right. Why is there a topic in the play we just was in the Yes. So, what is this? How do you think of Alex? I'm not sure. No, no, no, no, I'm not sure. How are you, Shrania? Yes, I'm fine. Okay, Alex. So, this is Alex, Klaacinke. We can show you some videos. We have some videos and now we will put in the book one. And we will put in the book three. We will put in the book two and so on. And I have this structure and a stack. I have a principle about this element that I have seen. So, I can make a two-way. So, I would like to show you all the things. If you have a first-in, last-in, first-out, first-in, last-outs. That's the first-out. So, that's the first-out. So, I know that. I can show you some elements. I can show you some elements that I have seen before. So, that's it. I can show you some elements. And a stack is quite a nice structure. When you can help us with some elements that I have seen in the past. I don't have any more intuitive elements. I think that for some examples I have seen for some elements. And I have not been able to see what the first-out means. What the first-out needs is. And a stack is really just one structure. So, we are implementing the same. We are trying to implement the same stack. We are going to create a new one. We are going to create a new one. And we are going to create some projects. Some programs that I have seen. And we are going to show you some projects. And now, the first-out, at the beginning, is the index minus one, which means that the stack is working. Because we are going to create a new one. And we are going to create a new one. We are going to create a new one. And we are going to create a new one. For example, the next-out is the index. The next-out is the stack. The first-out is the first-out, and the first-out is the first-out. And we are going to create a new one. And we are going to create a new index. And we are going to create a new index. Yes, the same. And when the first-out is done, I can only create a new one. And that is the whole implementation of the stack. So, I don't have anything special. So, now, I want to create a new one. I, in principle, don't have to be in a closed place, I can only say that now the number of shows is new. What does it mean? At least I have only one number. When I see the third-out, the number of shows here. And now, when I see the second-out, the number of shows here. That is the stack. Now, when I want to do something good, I will only add one number to the number of small ones. So, I will say this. To make it more clear, everything is ready, the stack is ready, the number of small ones is ready. So, we can create this one. And we will also create a stack. And we will also create a stack. So, we will create a structure, a stack, but we will be able to create a stack, and we will be able to create a new one. And, in the end, the third-out is expected to be developed, and the first thing is that we need to create a stack, and we will be able to create a stack, and we will be able to create a new one. And we will be able to create a stack. Yes, the first thing is that you can create a stack. I am very happy that I have shown a stack in the first place. I can show how you closet and something in there. And I will tell you that hayır An older than the next element. For the block top. There are many items in there, I will explain a cat behind the cards. I'm going to write this in the first step. I'm going to write this in the first step. I'm going to write this in the first step. I'm going to write this in the first step. I'm going to write this in the first step. I'm going to write this in the first step. I'm going to write this in the first step. I'm going to write this in the first step. I'm going to write this in the first step. I'm going to write this in the first step. I'm going to write this in the first step. I'm going to write this in the first step. I'm going to write this in the first step. I'm going to write this in the first step. I'm going to write this in the first step. I'm going to write this in the first step. I'm going to write this in the first step. There are some questions. Let's try to draw one model well prepared by us, let's try to draw a puzzle, let's not try to draw a puzzle or a puzzle, let's not try to draw a puzzle or a puzzle, let's not... Let's do this, we're going to be going ahead. me, me, me, me... Well, let's do things that go forward. uma Malaysia. uma Malaysia. II, me, me, me, I look, I may be going ahead. 5,000,000,000,000,000. Zij nationality. We'll have more research there than here. first we'll have more such projects there. first... I'm not sure if I'm wrong, but I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong. I'm not sure if I'm wrong."
    smaller_meeting = " Okay, thank you. This is a meeting of the RMS Springfield for July the 16th of 2024 at exactly 6 p.m. call on the meeting to order. We will do introduction here. I'm Mayor Patrick Terry for the RMS Springfield. We have Christie Grunheid at assistant CEOs at the controls. Our CEO calling Draper is present. Deputy Mayor to my right in the sending order in Council Ward 1 is Glen Fuel. Council Ward 2 is Andy Kaczynski. Council Ward 3 is Mark Miller. Council Ward 4 is Melinda Warren. Melinda if you want to do invocation. May we all enter into the meeting with open ears and open minds sharing our knowledge and experience while working together for the betterment of our municipality. Thank you very much. Then I'll do the land acknowledgement. The RMS Springfield acknowledges that we are gathered on an assessor lands. Treaty one territory, traditional territory, the initiative. Treaty three, OGC, Dakota, Dene, people on the national homeland and on the national homeland of the Red River, Métis. That is item number three and item number four is approval of the agenda. Can I move her in a second for that please? Melinda and Glen. Any questions, additions or modifications to the agenda? Mark? Close meeting. Okay. Anything else? I see none. Then I can I get a show of hands that those in support of the approval of the agenda? Anonymous and is carried. Move her in a seconder for the adoption of the minutes. Patrick and Andy. Any questions? Any questions, additions from Council adoption of the minutes for July 2nd? I see none. Show of hands. Those in support and is carried. We have a question period. Nobody is in front of us here. We have 50 minutes allocated. The only question pertaining to the current agenda online. I see nobody online as well. So at this point here, we'll carry on to consent agenda. That's item number 10. Oh, sorry. Tammy, did you have anything for question period at all? I just saw you coming in. Okay. Then what we'll do will get to consent agenda that's item number 10. I have a comments for 10.2 and 10.3. So I'm not sure if we can actually do the consent agenda there. So if I can do the first one 10.1, the AMM news bulletin, kind of move her in a seconder for that, please. Mark and Glenn. Is there any questions with regards to that from Council? I see none. I can get a show of hands of those in support. Anonymous and is carried. Next one is a 10.2. That's stars. Can I get a move in a sector for that? Patrick and Glenn. I had a comment here there. I am a big proponent for stars. This is a vital component for our health care. I've had first first hand experience with stars in Manitoba Air Ambulances in the North. Many communities in the North rely on air ambiance provided by the government to either via airport telecopter. And air ambiance, a propeller or a jet. When there's babies involved in adults alike, this is an extremely good tool for the health department. It is utilized in the Army Springfield. And that's all I had to say with regards to that. I'm kind of get a mover or a show of hands of those in support of stars. As unanimous and is carried. Then if we get to the thank you card, the SCI grad recipients and move in a sector for 10.3. Please. Melinda and Andy. At this point here, we have the card from Leila plant there. We increase the amount to $750 from 500, and which I think is was an exceptional idea. From our deputy mayor to increase that. Because most of the honorary burst reason so on are at that 750 level. And I'm not saying that we have to match costs there, but it was really well, well received. The SCI staff principal Kevin Dell, extremely large crowd present as every year. And their families at the club region, Casino, which is an impressive facility. But still not in our community. So it'll be nice to have the grad eventually be here at the our new recreational facility, which requires our new water treatment plant to be able to operate and provide our community going forward. So appreciate the card from from Leila is very well done. Any other questions or comments from council at all. I see none. Can I get a show hands of those in support of 10.3. Thank you, Kurt. So unanimous and is carried. We'll get to a new business there. And that's the purchase of the asphalt pot box trailer, mover and a seconder for that, please. Linda and Andy. Any questions from council with regards to this. You have a hand up, honey. Yes, Mr. Mayor. I would like to ask if we purchasing ourselves or we. Using that company that helping a purchase equipment through AMM. Or can you. Yeah. Did they can help us with purchasing equipment? We never use them like purchasing the trucks or a fire department chassis and. Police car. Yeah. I think that the industrial machine. Anc is part of that canoe procurement. They have to be a listed supplier. So maybe blame fear able to confirm if that's an opportunity through canoe procurement or not. Yeah, yeah, we. We did look into seeing if that was a listed company. And I don't believe they were so. I'm not 100% sure if there is any advantages for the news belly for canoe in this in this equipment purchase. So. Okay, thank you. Any other questions from council at all. I see none. Then I can read the resolution. It was all the council of the arm a springfield approves the purchase of one new asphalt hot box trailer from industrial machine. Not to exceed the budgeted amount of $155,000 including taxes with funds coming from the vehicle and equipment reserve. I get a show hands of first in support of purchase of asphalt hot box trailer. That's that's unanimous and it is carried. Then we'll get to 11.2 purchase of rehab trailer or specifically rehabilitation trailer for the fire department. Move on a seconder for that place Patrick and Glenn. The result of the council of the arm a springfield approves the purchase of one new rehabilitation trailer from pro pack for $59,630.27 US plus applicable taxes, transport fees and broker fees with funds to come from the vehicle and equipment reserve. Are we able to amend the resolution to show Canadian funds beside it there. I think it's $80,000. It's $59,630.27 US. When we write the check we have to get the exact US amount on that day of the check rating. Oh, so that will change. It could change from what's being proposed right now Canadian. So. Okay. All right. I understand that. Can I get a show any questions from council with regards to that? I see none. Can I get a show hands of those in support of the purchase of rehabilitation trailer? That is unanimous and it is carried 11.3 that's financial statements. Move on a seconder. Please. Well, Linda Mark. I had one question with regards to financial statements. It says here basically regards the notes to financial statements of may 3124. The tax-living grants and Lou have been collected in which accounts for our deficit. Can we have some clarification on this so there's there's no misinformation with regards to that. So we're still waiting for the tax levy and the grants to come through. Are they delayed for any reasons? Okay. So this is retained and it's not different to this year. Any other questions from council at all? I see none. Then can I get a show hands of financial statements those in support? That is unanimous and it is carried 11.4 list of accounts. I move in a seconder for that. Please. Thank you. That's a Melinda England. Whereas disbursements have been reviewed for the period of June 7th to July 5th, 2024, be it resolved that all accounts listed on the attached printout from check 160442 to 160721 totaling 2 million 88,0001 2756 be approved for payment as well as the following EFT entries. Western financial 15,000 1 27 68 Amelia 358 355 super past 177 59 John Deere 8,000 4448 91 6,000 5 76 83 6,000 5 76 83 and 454429. RBC visa 17,000 7 3069 Manitoba Hydro 30,000 3904 Bell mobility 1994 37 Bell Canada 6,000 1 31 72 Volvo 8,000 831 84 and 12,000 67 14 and be it further resolved that June payroll on the amount of 373 5733 be approved. Any questions comments from council. Mark. Yeah, we collect points for all our purchases. Any other questions from council. I see none can I get a show of hands those in support to lists of accounts under 11.4. It is unanimous and is carried. If we get to 11.5 it's a nonprofit tax cancellation center eyes performing arts center of excellence or space. Move her in a seconder for that to please Patrick and Melinda. Beers all the council of the arm a Springfield approved a nonprofit organization grant for Springfield performing arts center of excellence in the amount of $7,909 and 19 cents. I have a question about that the taxes are 6957 plus the rears so that would take us to ballpark of 7100. But we're looking at $7900 for that. I'm sure what the math is there. So the tax bill that council is seeing a copy of those are rears are not to date so that was when the original bill was printed in July of 20. 23. Oh I see so the 140 is the still of standing so that's the 140. Okay, but that's still there's monthly penalties that have been accruing since that time. Oh, so that's where the 79 is from. Yeah. Okay. Any questions from council for guards to that. Yes, I would like to ask if they always paying penalties they never come up with it or this is just the first time. I think it's a good question. Yeah, so they were kind of in a strange situation where they were also renting additional space in the mall and own this property as well. So they had come to council for a council cancellation of the property taxes at that time council didn't agree with the cancellation because the activities weren't taking place in the building on Main Street. You know, gotten rid of their space and the mall moved back to that building they've come back to council and now it's council's opportunity to approve or deny again. I know, but that's not what I ask I ask you if they always on the rears. No, they're not only in a rare. Okay. No, this is the first time. Thank you. And as council we've had discussions about this in camera and then out of camera there, but it's to the point there would be no further assistance in my opinion. I got one vote there, but I think hopefully this this helps there, but we have to take that in consideration there. Any other comments from council? See none. Can I get a show hands of those in support of the nonprofit tax as in cancellation space? The unanimous is carried. I'll get to 11.6 letter of support for a john q built for collaborative efficiency advocate position. Can I get a mover in a second with regards to that please. Melinda and Glenn. Any questions from council with regards to a letter of support? Andy? Mr. Mayor, I want to ask like they suggest that we use this efficiency money to buy. We can we can go directly ourselves to efficiency money to buy. We don't have to go through john q. This is john q asking that they're going to go align with efficiency manitoba. They've got a grant for 120,000 in the john q will likely match that to a certain level. Yes, we could do that. And so anybody can use efficiency manitoba. But we're linking with efficiency manitoba, the government entity there to get that special environmental incentive as well. So we're going to have a $5% MECD. I'm trying to remember with that acronym stands for but it'll the the structure are john q daycares are built better than the manitoba environmental committee designations there. I believe that's what MECD stands for or something with regards to that. So M a efficiency manitoba and john q with their daycare program is is better than the what the government is looking for. And efficiency manitoba will work with us to do even a better job. And this is a lining directly with them for for our john q built in john to john q a daycare. So yeah, they anybody can you can can as a small business and so on use them. But john q has been using them and now they're looking to to have a direct contact with them to to make the initiatives that much better. And just asking for support with regards to the green initiatives that efficiency manitoba will will give to them as well. Because we're part of that group of the I want to take much Paulton region there they're throwing it out to surrounding areas city when it. And just then but we kind of question what john q did for us over the years and they didn't do actually absolutely nothing till now. So we paying the you know the bill for them like I don't know $200,000 a year and what do we getting in the returning. Well, we're I don't know if it's it's a one time payment what I understand is $20,000 there. And like daycare that 39 acres of land that we we bought there, which was a very prune to good move on councils part there because that land is is only going up in value. But we have no land for a daycare. This community needs a daycare as a high priority for allowing us to go forward there because the daycare is huge and just being a grandparent and with my with my sons and daughter laws and my grandkids. It's a huge cost and location location location is is is paramount. And if we have this daycare like they have an eSalkirk lorette headly in the city of Winnipeg there's 20 plus these daycares around there they're very efficient. There's actually a university I know the name but I'm not going to mention the name that is doing a study with regards to our our business initiative for that because it's it's a global project that's getting a lot of insight from from other entities there in the energy efficiency coming from these buildings the costs are our stream line and not to mention the environmental green initiative is is top. So we are getting lots from this and john q we will approach them I hope once we get everything ready or water treatment plant up and running there. To allow our community to go forward and daycare is something this community needs. Nobody say that we don't need it but not necessarily the land we purchase going to go there we have land that we want to build a. What a treatment plant and we didn't pay you know absorbent amount of money for that land and we could have used that one for that purpose to so it's not only that just because we purchased the land just. It's you know the bill for for that daycare center we nobody say that we don't need the daycare center but you know. We we have we we had the land to to put the daycare center before that thank you thank you any other questions from council at all. And if I get a vote or can I get the resolution be resolved that the mayor be authorized to sign a letter of support for jq bill for a collaborative advocate position on behalf of the armistro. Springfield I got to show hands of those in support of a letter of support for john q. That is a Melinda Glenn and Patrick those post mark and Andy. And I'm going to see if I carried. 11.7 amount of municipalities administrators review and revision 11.7 I'm moving a seconder for that to please Melinda and Patrick. Any questions from council with regards to that. I can get the resolution read. Did you have a question any. Who structured this letter to us. Manitoba municipality of administrators. Correct. Yeah. And what what can you tell us about this organization. Well, the MMA so municipal municipal administrators is kind of my AMM to council. So it's the organization I belong to and I have to get my professional credits through to keep a good standing with them. So it's this group that was asked to put. There, you know, comments regarding the municipal board mandate role function and practices. So they put together resolution and sent it out to all the members and ask that we bring it forward to council for review. And if you want to vote on the resolution and provide it. That's an option. So they're stating here. Municipality and wasteful processes and their interaction municipal boards. So this have something to do with built 37 to. Oh, well, I think the province is doing a review of the municipal municipal board mandate. So they've reached out to certain groups to comment. And they reached out to MMA to comment on this. This is some stick to do with the built 37. I don't know specifically. I don't think it does. But I can confirm, but. The municipal board follows falls under a different act. So. So it's a specific to the municipal board. Thank you. Any other questions council? Have you read that resolution yet? No. Do we have to read it allowed or can we agree to that we. Let's read the therefore be a result section. Therefore be it resolved that the AMM lobby, the province of Manitoba to undertake a comprehensive review of the mandate role and function. And municipal board to evaluate its relevance and actual value in today's municipal sector. And further that the province of Manitoba complete a third party value for money service delivery review of the municipal board processes and undertake process improvements. To streamline functions, reduce red tape and reduce municipal costs. And further that the province of Manitoba engage AMM and MMA to participate as key stakeholders in the preparation of terms of reference for these efforts. And any steering or oversight body for this work. With the resolution read, can I get a show hands of those in support to the amount of municipal administrators review and revision. As unanimous and is carried 11.8 council schedule schedule. Can I get a mover and a seconder Patrick Melinda? Be it resolved that the following changes be approved to the council meeting schedule Tuesday, October 8th, Committee of the whole cancelled. Any questions council. I see none any. And why we counseling for the reason. Council just approved the planning session with council and staff on October 7th to 9th. Thank you. I got a show hands those in support to the council schedule 11.8. It's unanimous and sorry Andy do you have your hand up. Yeah, that's unanimous and carried. Then we're prepared to close the meeting for in camera there. Can I get a mover and a seconder for that please. Melinda and Patrick. I'll be it resolved that this meeting recessed to encamer to discuss legal issues. And be it further resolved that all matters show remain confidential until report is made public. Thank you very much. It's 6.57 p.m. We're coming out of a closed meeting. Can I get a mover and a seconder and error to come out of the closed meeting. Melinda and Andy. And we've got nothing more on the agenda. Everything was discussed in camera. If we can adjourn the meeting and move her in a seconder for that please. Melinda and Patrick. Then the meeting is adjourned at six. Oh, show hands. unanimous. The meeting is adjourned at six. Thank you."

    something = "sadasdasdasdaf  asdadfqefwegwsd gwegweg wegweg qqqqqqqqfg gwegweg Count tokens using tiktoken if available; otherwise fall back to word count. Count tokens using tiktoken if available; otherwise fall back to word count. Count tokens using tiktoken if available; otherwise fall back to word count. i dont asd asd asd w w sqqqqw wrt qw www ggg wwww wq wqr ffffasfasfa saf asfas sf w ww wwwwww Count tokens using tiktoken if available; otherwise fall back to word count. i dont asd asd asd w w sqqqqw wrt qw www ggg wwww wq wqr ffffasfasfa saf asfas sf w ww wwwwww Count tokens using tiktoken if available; otherwise fall back to word count. i dont asd asd asd w w sqqqqw wrt qw www ggg wwww wq wqr ffffasfasfa saf asfas sf w ww wwwwww Count tokens using tiktoken if available; otherwise fall back to word count. i dont asd asd asd w w sqqqqw wrt qw www ggg wwww wq wqr ffffasfasfa saf asfas sf w ww wwwwww Count tokens using tiktoken if available; otherwise fall back to word count. i dont asd asd asd w w sqqqqw wrt qw www ggg wwww wq wqr ffffasfasfa saf asfas sf w ww wwwwww Count tokens using tiktoken if available; otherwise fall back to word count. i dont asd asd asd w w sqqqqw wrt qw www ggg wwww wq wqr ffffasfasfa saf asfas sf w ww wwwwww Count tokens using tiktoken if available; otherwise fall back to word count. i dont asd asd asd w w sqqqqw wrt qw www ggg wwww wq wqr ffffasfasfa saf asfas sf w ww wwwwww"
    summarizer = Summarizer()
    res = summarizer._count_tokens(something)
    print(res)
    print(summarizer._split_text_into_chunks(something, summarizer.CHUNK_TOKEN_SIZE))
