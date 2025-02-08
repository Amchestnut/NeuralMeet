import pyaudio
import wave
import time
import whisper
import os
import sys
import threading
from queue import Queue
from Summarizer import Summarizer

class RealtimeTranscriber:
    """
    Continuously captures audio from the microphone and enqueues audio chunks.
    A separate processing thread transcribes and summarizes the audio in real time,
    so that recording is not blocked by processing delays.
    """
    def __init__(
        self,
        output_file="realtime_summary.txt",
        model_size="base",
        chunk_sec=5,
        summarize_interval_sec=60,
        file_type="meeting"
    ):
        self.stop_event = threading.Event()
        self.audio_queue = Queue()
        self.output_file = output_file
        self.chunk_sec = chunk_sec
        self.summarize_interval_sec = summarize_interval_sec

        print(f"Loading Whisper model '{model_size}' ...")
        self.model = whisper.load_model(model_size)
        self.summarizer = Summarizer(file_type=file_type)

        self._transcript_buffer = []  # stores transcribed text for each chunk
        self._time_accumulator = 0.0  # seconds accumulated since last summary
        self._context_summary = file_type  # initial context is simply the file_type
        self._minute_count = 0  # counts 1-minute intervals

        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def _record_chunk(self, stream, chunk_size, rate):
        """
        Record self.chunk_sec seconds of audio (using multiple small reads) and return the raw frames.
        """
        frames = []
        num_of_reads = int(rate / chunk_size * self.chunk_sec)
        for _ in range(num_of_reads):
            if self.stop_event.is_set():
                break
            try:
                data = stream.read(chunk_size, exception_on_overflow=False)
            except Exception as e:
                print("Recording error:", e)
                break
            frames.append(data)
        return b''.join(frames)

    def record_audio(self):
        """
        Producer thread: continuously record audio and put each chunk on the queue.
        """
        chunk_size = 1024
        rate = 16000
        channels = 1

        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=chunk_size)
        print("Recording thread started.")
        while not self.stop_event.is_set():
            frames = self._record_chunk(stream, chunk_size, rate)
            if frames:
                self.audio_queue.put(frames)
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Recording thread stopped.")

    def process_audio(self):
        """
        Consumer thread: continuously pull audio chunks from the queue, transcribe them,
        and accumulate the transcription. When the accumulated time exceeds the summarization
        interval, perform a summary and update the context.
        """
        channels = 1
        rate = 16000
        while not self.stop_event.is_set() or not self.audio_queue.empty():
            try:
                frames = self.audio_queue.get(timeout=1)
            except Exception:
                continue

            text = self._transcribe_chunk(frames, channels, rate)
            print(f"[Partial text]: {text}")
            self._transcript_buffer.append(text)
            self._time_accumulator += self.chunk_sec

            # If we've accumulated enough time, process a summary.
            if self._time_accumulator >= self.summarize_interval_sec:
                self._minute_count += 1
                minute_text = "\n".join(self._transcript_buffer)
                chunk_part, updated_context = self._summarize_minute(minute_text)

                print(f"\n--- Minute {self._minute_count} Summary ---")
                print(chunk_part)
                print("--- End Summary ---\n")

                with open(self.output_file, "a", encoding="utf-8") as f:
                    f.write(f"\n## Minute {self._minute_count}\n")
                    f.write(f"**Context before chunk**: {self._context_summary}\n\n")
                    f.write(f"**New Summary**:\n{chunk_part}\n\n")

                self._context_summary = updated_context
                self._transcript_buffer = []
                self._time_accumulator = 0.0

            self.audio_queue.task_done()
        print("Processing thread stopped.")

    def _transcribe_chunk(self, frames, channels=1, rate=16000):
        """
        Write the recorded frames to a temporary WAV file and transcribe it using Whisper.
        """
        temp_wav = "temp_chunk.wav"
        wf = wave.open(temp_wav, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # pyaudio.paInt16 = 2 bytes
        wf.setframerate(rate)
        wf.writeframes(frames)
        wf.close()
        result = self.model.transcribe(temp_wav)
        text = result["text"].strip()
        return text

    def _summarize_minute(self, minute_text):
        """
        Use the summarizer to process the minute's transcript and update context.
        """
        chunk_summary, updated_context = self.summarizer._process_chunk(
            chunk_text=minute_text,
            context_summary=self._context_summary
        )
        return chunk_summary, updated_context

    def run(self):
        rec_thread = threading.Thread(target=self.record_audio, daemon=True)
        proc_thread = threading.Thread(target=self.process_audio, daemon=True)
        rec_thread.start()
        proc_thread.start()

        print("Press Ctrl+C to stop.")
        try:
            while not self.stop_event.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("KeyboardInterrupt received. Stopping...")
            self.stop_event.set()

        rec_thread.join()
        proc_thread.join()
        print(f"All done. Summaries appended to: {self.output_file}")
        sys.exit(0)

if __name__ == "__main__":
    rt = RealtimeTranscriber(
        output_file="realtime_summary.txt",
        model_size="base",
        chunk_sec=5,
        summarize_interval_sec=60,
        file_type="meeting"
    )
    rt.run()
