from speech_to_text.audio_file_transcriber import AudioFileTranscriber
from ai_model.summarizer import Summarizer

class AudioFileProcessor:
    """
    Handles chunk-based processing of a full audio/video file.
    1) Transcribe the file into a complete transcript.
    2) Split the transcript into chunks.
    3) Summarize each chunk with rolling context.
    4) Generate the final summary.
    """

    CHUNK_TOKEN_SIZE = 3000  # The maximum number of tokens per chunk

    def __init__(self, model_size="medium"):
        self.transcriber = AudioFileTranscriber(model_size=model_size)
        self.summarizer = Summarizer()

    def _split_text_into_chunks(self, text, max_tokens):
        """
        Splits the full transcript into chunks so that each chunk stays under `max_tokens`.
        Simple approach here is word-based counting. If you prefer more precise token counts,
        you can integrate the summarizer's _count_tokens method.
        """
        chunks = []
        words = text.split()
        current_chunk = []
        current_chunk_tokens = 0

        for word in words:
            # For a simple approach, we treat 1 word = 1 token
            token_count = 1
            if (current_chunk_tokens + token_count) > max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_chunk_tokens = token_count
            else:
                current_chunk.append(word)
                current_chunk_tokens += token_count

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def process_file(self, file_path, file_type):
        """
        Main entry:
          1) Transcribe the entire file
          2) Split into chunks
          3) Summarize each chunk in sequence
          4) Possibly reduce the chunk summaries if they exceed final token limits
          5) Generate and save the final summary
        """
        print(f"Transcribing the {file_type} audio/video file: {file_path}")
        full_transcript = self.transcriber.transcribe_audio(file_path)

        # Save the transcript to a file for reference
        with open("transcript.txt", "w", encoding="utf-8") as f:
            f.write(full_transcript)

        print("Splitting the transcript into chunks...")
        chunks = self._split_text_into_chunks(full_transcript, self.CHUNK_TOKEN_SIZE)
        print(f"Total chunks: {len(chunks)}")

        context_summary = ""
        history_list = []

        for i, chunk_text in enumerate(chunks, 1):
            print(f"Processing chunk {i}/{len(chunks)}...")
            chunk_part, updated_context = self.summarizer.process_chunk(
                chunk_text,
                context_summary,
                file_type
            )
            print("CHUNK_PART:")
            print(chunk_part)
            history_list.append(chunk_part)
            context_summary = updated_context

        # Check combined token count
        combined_tokens = self.summarizer._count_tokens(" ".join(history_list))
        print(f"Combined history token count: {combined_tokens}")

        # If history is too large, reduce it
        if combined_tokens > self.summarizer.FINAL_SUMMARY_THRESHOLD:
            print("Reducing history summaries to fit token limits...")
            history_list = self.summarizer.reduce_history(history_list)

        print("Generating final summary...")
        final_summary = self.summarizer.final_summary(history_list, file_type)

        # Save the final summary
        output_file = f"output/final_{file_type}_summary.txt"
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(final_summary)
            print(f"Final {file_type} summary saved to: {output_file}")
        except Exception as e:
            print("Error writing the final summary file:", e)

        return final_summary
