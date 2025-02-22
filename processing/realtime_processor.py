import time
import sys
import threading
from ai_model.summarizer import Summarizer
from speech_to_text.realtime_transcriber import RealtimeTranscriber

class RealtimeProcessor:
    """
    Manages real-time audio capture and summarization:
    1) Uses RealtimeTranscriber for capturing mic audio in chunks.
    2) In a consumer thread, transcribes each chunk and appends text to a buffer.
    3) Periodically calls Summarizer on the accumulated text to produce a rolling summary.
    """

    def __init__(
        self,
        summarize_interval_sec=60,
        chunk_sec=5,
        model_size="medium",
        file_type="meeting",
        output_file="realtime_summary.txt"
    ):
        self.stop_event = threading.Event()
        self.summarize_interval_sec = summarize_interval_sec
        self.file_type = file_type
        self.output_file = output_file

        self.transcriber = RealtimeTranscriber(
            model_size=model_size,
            chunk_sec=chunk_sec
        )
        self.summarizer = Summarizer()

        self.transcript_buffer = []         # collects transcribed text from chunks
        self.time_accumulator = 0.0         # tracks seconds since last summary
        self.context_summary = file_type    # initial context is simply the file type
        self.minute_count = 0

        self.consumer_thread = threading.Thread(target=self._process_audio, daemon=True)

    def _process_audio(self):
        """
        Consumer thread: pull audio chunks from the transcriber queue, transcribe them,
        and accumulate the resulting text. When enough time has passed, summarize.
        """
        while not self.stop_event.is_set() or not self.transcriber.audio_queue.empty():
            try:
                frames = self.transcriber.audio_queue.get(timeout=1)
            except:
                continue

            # Transcribe chunk
            text = self.transcriber.transcribe_chunk(frames)
            print(f"[Partial text]: {text}")
            self.transcript_buffer.append(text)
            self.time_accumulator += self.transcriber.chunk_sec

            # If we've accumulated enough time, do a summary
            if self.time_accumulator >= self.summarize_interval_sec:
                self.minute_count += 1
                combined_text = "\n".join(self.transcript_buffer)
                chunk_part, updated_context = self.summarizer.process_chunk(
                    combined_text,
                    self.context_summary,
                    self.file_type
                )

                print(f"\n--- Minute {self.minute_count} Summary ---")
                print(chunk_part)
                print("--- End Summary ---\n")

                # Append to an output file
                with open(self.output_file, "a", encoding="utf-8") as f:
                    f.write(f"\n## Minute {self.minute_count}\n")
                    f.write(f"**Context before chunk**: {self.context_summary}\n\n")
                    f.write(f"**New Summary**:\n{chunk_part}\n\n")

                # Reset
                self.context_summary = updated_context
                self.transcript_buffer = []
                self.time_accumulator = 0.0

            self.transcriber.audio_queue.task_done()

        print("Consumer thread stopped.")

    def start(self):
        """
        Start the producer (record_audio) and consumer (_process_audio) threads for real-time audio capture and then summarization.
        """
        rec_thread = threading.Thread(target=self.transcriber.record_audio, daemon=True)
        rec_thread.start()
        self.consumer_thread.start()

        print("Press Ctrl+C to stop real-time processing.")
        try:
            while not self.stop_event.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("KeyboardInterrupt received. Stopping...")
            self.stop_event.set()
            self.transcriber.stop()

        rec_thread.join()
        self.consumer_thread.join()
        print(f"All done. Summaries appended to: {self.output_file}")
        sys.exit(0)
