from llm.summarizer import Summarizer

summarizer = Summarizer()

def process_text(text: str, user_options: dict, rolling_context: str):
    """
    Process a transcript chunk using the Summarizer.

    Parameters:
      - text: the transcribed text for this chunk.
      - user_options: dict containing options, e.g. {"file_type": "meeting"}.
      - rolling_context: the current rolling context summary.

    Returns:
      - chunk_summary: the processed (summarized) text from this chunk.
      - updated_context: the updated rolling context.
    """
    # Get the file type from the user options (defaulting to "meeting")
    file_type = user_options.get("file_type", "meeting")

    # Process the chunk using the Summarizer
    chunk_summary, updated_context = summarizer.process_chunk(text, rolling_context, file_type)
    return chunk_summary, updated_context
