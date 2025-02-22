class PromptFactory:
    """
    All prompt templates are stored here.
    This helps keep Summarizer code cleaner, focusing only on the LLM interaction logic.
    """

    def __init__(self):
        # Chunk-level prompts: these instruct the LLM on how to summarize each chunk
        self.chunk_prompt_templates = {
            "meeting": (
                "System: You are a precise assistant that captures important details and presents them in a structured format.\n\n"
                "Meeting Transcript Chunk:\n{chunk_text}\n\n"
                "Current Context Summary:\n{context_summary}\n\n"
                "Instructions:\n"
                "1. Write `CHUNK_PART`:\n"
                "2. Provide a clear overview of what was discussed\n"
                "3. Focus on:\n"
                "   - Key decisions and their context\n"
                "   - Important discussion points\n"
                "   - Action items and next steps\n"
                "   - Notable concerns or issues\n"
                "4. Structure using Markdown:\n"
                "   - Use '## Topic' for main sections\n"
                "   - Use '* ' for key points\n"
                "   - Use '**bold**' for decisions\n"
                "   - Add '\\n' between sections\n"
                "5. Be clear but not overly detailed\n"
                "You can use up to 500 tokens to write a detailed structured overview with markdowns.\n"
                "6. Write a delimiter line `---DELIMITER---`\n"
                "7. Write `UPDATED_CONTEXT`:\n"
                "8. Write an updated context of the conversation.\n"
            ),
            "lecture": (
                "System: You are a precise assistant that captures important details and presents them in a structured format.\n\n"
                "Lecture Transcript Chunk:\n{chunk_text}\n\n"
                "Current Context Summary:\n{context_summary}\n\n"
                "Instructions:\n"
                "1. Write `CHUNK_PART`:\n"
                "2. Provide a clear overview of what was taught\n"
                "3. Focus on:\n"
                "   - Main concepts explained\n"
                "   - Any examples given\n"
                "   - Important definitions\n"
                "   - Core relationships between concepts\n"
                "4. Structure using Markdown:\n"
                "   - Use '## Topic' for main concepts\n"
                "   - Use '* ' for key points\n"
                "   - Use '**bold**' for important terms\n"
                "   - Add '\\n' between sections\n"
                "5. Be clear but not overly detailed\n"
                "You can use up to 500 tokens to write a detailed structured overview with markdowns.\n"
                "6. Write a delimiter line `---DELIMITER---`\n"
                "7. Write `UPDATED_CONTEXT`:\n"
                "8. Write an updated context of the conversation.\n"
            ),
            "call": (
                "System: You are a precise assistant that captures important details and presents them in a structured format.\n\n"
                "Phone Call Transcript Chunk:\n{chunk_text}\n\n"
                "Current Context Summary:\n{context_summary}\n\n"
                "Instructions:\n"
                "1. Write `CHUNK_PART`:\n"
                "2. Provide a clear overview of the conversation\n"
                "3. Focus on:\n"
                "   - Main topics discussed\n"
                "   - Important agreements or decisions\n"
                "   - Key requests or requirements\n"
                "   - Follow-up items\n"
                "4. Structure using Markdown:\n"
                "   - Use '## Overview' for the main section\n"
                "   - Use '* ' for key points\n"
                "   - Use '**bold**' for decisions/agreements\n"
                "   - Add '\\n' between sections\n"
                "5. Be clear but not overly detailed\n"
                "You can use up to 500 tokens to write a detailed structured overview with markdowns.\n"
                "6. Write a delimiter line `---DELIMITER---`\n"
                "7. Write `UPDATED_CONTEXT`:\n"
                "8. Write an updated context of the conversation.\n"
            )
        }

        # Final summary prompts: these instruct the LLM to combine chunk parts into a final organized report
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
                "   - Any important data related to money must be included and correct.\n"
                "4. Maintain a clear structure (use headings or bullet points) so it's easy to read.\n"
                "5. Use clear headings and bullet points (Markdown symbols like '*', '**', '#', '##') for readability.\n"
                "6. Do not output JSON—output plain text.\n"
                "7. Ensure that the final text is cohesive and complete.\n"
                "You may write an output even if it is 4000 tokens long.\n"
            ),
            "lecture": (
                "You are creating a final, detailed, concise and comprehensive **Lecture** report.\n"
                "This should capture key topics, explanations, highlight any examples that were given, and insights.\n\n"
                "Below are all the important information that was discussed:\n\n{combined_history}\n\n"
                "Instructions:\n"
                "1. Write a thorough but organized summary of the lecture material.\n"
                "2. Retain important details, focusing on concepts explained, examples given, and any critical definitions.\n"
                "3. Use headings or bullet points to structure the summary (e.g., # Topics, ## and *text* for Examples, use '**' for Key Points).\n"
                "4. Do **not** output JSON—only plain text.\n"
                "   Use symbols like '*', '**', '#', '##', '###' and other Markdown symbols so it's easily used in Obsidian, but focus mostly on '*', '**' and '#', '##'.\n"
                "5. Keep it cohesive, removing only obvious repetition.\n"
                "6. You must not lose any data, only organize the given data and highlight it better.\n"
                "You may write an output even if it is 4000 tokens long.\n"
            ),
            "call": (
                "You are creating a final, detailed, concise and comprehensive **Phone call** report.\n"
                "It should capture the key points of the conversation, follow-ups, and next steps.\n\n"
                "Below are all the important information that was discussed:\n\n{combined_history}\n\n"
                "Instructions:\n"
                "1. Provide a clear, organized recap of the phone call.\n"
                "2. Highlight important details:\n"
                "   - Key discussion items\n"
                "   - Agreements or commitments made\n"
                "   - Action items or due dates\n"
                "3. Do **not** output JSON—only plain text.\n"
                "   Use symbols like '*', '**', '#', '##', '###' and other Markdown symbols so it's easily used in Obsidian.\n"
                "4. Keep it cohesive, removing only obvious repetition.\n"
                "5. You must not lose any data, only organize the given data and highlight it better.\n"
                "You may write an output even if it is 4000 tokens long.\n"
            )
        }

    def get_chunk_prompt(self, file_type: str) -> str:
        return self.chunk_prompt_templates.get(file_type, self.chunk_prompt_templates["meeting"])

    def get_final_prompt(self, file_type: str) -> str:
        return self.final_prompt_templates.get(file_type, self.final_prompt_templates["meeting"])
