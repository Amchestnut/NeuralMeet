import soundcard as sc
import threading
import time
import requests
from queue import Queue

from .stt_client import transcribe_chunk_via_grpc

class SystemAudioProcessor:
    """
    Captures system playback audio (via WASAPI loopback) in 5-second chunks,
    sends them to the STT microservice, accumulates transcriptions, and every 60 seconds
    calls the LLM microservice to process the accumulated text.
    """
    def __init__(self,
                 stt_address="localhost:50051",
                 chunk_sec=5,
                 sample_rate=16000,
                 llm_endpoint="http://localhost:8001/process_text",
                 file_type="system_audio",
                 output_file="SystemAudio_realtime_summary.txt"):
        self.stt_address = stt_address
        self.chunk_sec = chunk_sec
        self.sample_rate = sample_rate
        self.llm_endpoint = llm_endpoint
        self.file_type = file_type
        self.output_file = output_file

        self.stop_event = threading.Event()
        self.audio_queue = Queue()

        # For accumulating transcriptions and summarization
        self.transcript_buffer = []      # collects transcribed text chunks
        self.time_accumulator = 0.0        # seconds of audio accumulated
        self.summarize_interval_sec = 60   # call LLM every 60 seconds of audio
        self.context_summary = file_type   # initial context is the file type
        self.minute_count = 0

    def start(self):
        record_thread = threading.Thread(target=self._record_loop, daemon=True)
        record_thread.start()

        print("Recording system audio... Press Ctrl+C to stop.\n")
        try:
            self._process_audio_chunks()
        except KeyboardInterrupt:
            print("KeyboardInterrupt received. Stopping...")
            self.stop()
        record_thread.join()

    def stop(self):
        self.stop_event.set()

    def _record_loop(self):
        # Use the default loopback microphone for capturing system audio.
        loopback_mic = sc.default_microphone()
        print(f"Capturing system audio from: {loopback_mic.name}")
        with loopback_mic.recorder(samplerate=self.sample_rate,
                                   channels=1,
                                   blocksize=1024) as recorder:
            frames = []

            # For float32, each sample is 4 bytes.
            total_bytes_needed = self.sample_rate * 4 * self.chunk_sec

            while not self.stop_event.is_set():
                # Record a small block of frames (1024 frames per call)
                data = recorder.record(numframes=1024)

                # Convert the numpy array to bytes (float32 -> 4 bytes per sample)
                data_bytes = data.tobytes()
                frames.append(data_bytes)

                # Check if we have enough bytes for chunk_sec seconds of audio
                if sum(len(x) for x in frames) >= total_bytes_needed:
                    # Combine the bytes and trim if necessary
                    chunk_data = b"".join(frames)[:total_bytes_needed]
                    self.audio_queue.put(chunk_data)
                    frames = []  # reset for the next chunk
        print("[_record_loop] Stopped recording system audio.")

    def _process_audio_chunks(self):
        while not self.stop_event.is_set():
            try:
                chunk_data = self.audio_queue.get(timeout=1.0)
            except Exception:
                continue

            # Transcribe the current chunk using the STT microservice.
            transcription = transcribe_chunk_via_grpc(
                audio_chunk=chunk_data,
                stt_address=self.stt_address
            )
            print(f"[System Audio chunk] {transcription}")

            # Accumulate transcription and elapsed time.
            self.transcript_buffer.append(transcription)
            self.time_accumulator += self.chunk_sec

            # When enough audio has been accumulated, call the LLM microservice.
            if self.time_accumulator >= self.summarize_interval_sec:
                self.minute_count += 1
                full_text = "\n".join(self.transcript_buffer)
                payload = {
                    "text": full_text,
                    "user_options": {"file_type": self.file_type},
                    "rolling_context": self.context_summary
                }
                try:
                    resp = requests.post(self.llm_endpoint, json=payload)
                    resp.raise_for_status()
                    data = resp.json()

                    # Expecting a JSON response with "chunk_summary" and optionally "updated_context".
                    chunk_summary = data.get("chunk_summary", full_text)
                    updated_context = data.get("updated_context", self.context_summary)
                except Exception as e:
                    print(f"Error calling LLM service: {e}")
                    chunk_summary = full_text
                    updated_context = self.context_summary

                print(f"\n--- Minute {self.minute_count} Summary ---")
                print(chunk_summary)
                print("--- End Summary ---\n")

                # Append the summary to the output file.
                with open(self.output_file, "a", encoding="utf-8") as f:
                    f.write(f"\n## Minute {self.minute_count}\n")
                    f.write(f"**New Summary**:\n{chunk_summary}\n\n")

                # Reset the accumulation for the next interval.
                self.context_summary = updated_context
                self.transcript_buffer = []
                self.time_accumulator = 0.0

            self.audio_queue.task_done()
