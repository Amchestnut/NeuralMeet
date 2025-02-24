import time
import sys
import threading
import os

from old.ai_model.summarizer import Summarizer
from old.speech_to_text.system_audio_transcriber import SystemAudioTranscriber

class SystemAudioProcessor:
    """
    Manages real-time system audio (from FFmpeg pipe) and summarization:
      1) SystemAudioTranscriber reads audio chunks from sys.stdin.
      2) This processor pulls chunks from the transcriber.queue, transcribes them,
         and accumulates the text.
      3) Periodically calls Summarizer on the accumulated text to produce a rolling summary.
    """

    def __init__(
        self,
        summarize_interval_sec=60,
        chunk_sec=5,
        model_size="medium",
        file_type="meeting",
        output_file="system_audio_realtime_summary.md"
    ):
        self.stop_event = threading.Event()
        self.summarize_interval_sec = summarize_interval_sec
        self.file_type = file_type
        self.output_file = os.path.expanduser(output_file)

        # Ensure directory exists
        output_dir = os.path.dirname(self.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        self.transcriber = SystemAudioTranscriber(model_size=model_size, chunk_sec=chunk_sec)
        self.summarizer = Summarizer()

        self.transcript_buffer = []
        self.time_accumulator = 0.0
        self.context_summary = file_type
        self.minute_count = 0

        # We'll read from the transcriber's queue in a consumer thread
        self.consumer_thread = threading.Thread(target=self._process_audio, daemon=True)

    def _process_audio(self):
        """
        Continuously pull chunk frames from the transcriber queue,
        transcribe them using the same transcriber, and produce partial text.
        Accumulate partial text, and every X seconds do a chunk summary.
        """
        while not self.stop_event.is_set():
            # Wait for next chunk from the queue
            try:
                frames = self.transcriber.audio_queue.get(timeout=1)
            except:
                continue

            text = self.transcriber.transcribe_chunk(frames)
            print(f"[Partial from system audio]: {text}")
            self.transcript_buffer.append(text)

            self.time_accumulator += self.transcriber.chunk_sec
            if self.time_accumulator >= self.summarize_interval_sec:
                self.minute_count += 1
                combined_text = "\n".join(self.transcript_buffer)
                chunk_part, updated_context = self.summarizer.process_chunk(
                    combined_text,
                    self.context_summary,
                    self.file_type
                )

                print(f"\n--- Minute {self.minute_count} System Audio Summary ---")
                print(chunk_part)
                print("--- End Summary ---\n")

                # Append to the output file
                with open(self.output_file, "a", encoding="utf-8") as f:
                    f.write(f"\n## Minute {self.minute_count}\n")
                    f.write(f"**New Summary**:\n{chunk_part}\n\n")

                # Reset
                self.context_summary = updated_context
                self.transcript_buffer = []
                self.time_accumulator = 0.0

            self.transcriber.audio_queue.task_done()

        print("Consumer thread stopped.")

    def start(self):
        """
        1) Start the transcriber reading from stdin in a background thread.
        2) Start the consumer thread that processes each chunk and runs summarization.
        """
        print("Starting SystemAudioTranscriber...")
        self.transcriber.start_recording()
        self.consumer_thread.start()

        print("Press Ctrl+C to stop system audio processing.")
        try:
            while not self.stop_event.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("KeyboardInterrupt received. Stopping...")
            self.stop_event.set()
            self.transcriber.stop()

        self.consumer_thread.join()
        print(f"All done. Summaries appended to: {self.output_file}")
        sys.exit(0)
