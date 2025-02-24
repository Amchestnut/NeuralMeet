import pyaudio
import threading
import time
import requests
from queue import Queue

from .stt_client import transcribe_chunk_via_grpc

class MicrophoneProcessor:
    def __init__(self,
                 stt_address="localhost:50051",
                 chunk_sec=5,
                 sample_rate=16000,
                 channels=1,
                 llm_endpoint="http://localhost:8001/process_text",
                 file_type="meeting",
                 output_file="realtime_summary.txt"):
        self.stt_address = stt_address
        self.chunk_sec = chunk_sec
        self.sample_rate = sample_rate
        self.channels = channels
        self.llm_endpoint = llm_endpoint
        self.file_type = file_type
        self.output_file = output_file

        self.stop_event = threading.Event()
        self.audio_queue = Queue()

        # PyAudio settings
        self.p = pyaudio.PyAudio()
        self.chunk_size = 1024  # frames per buffer
        self.stream = None

        # For accumulating transcriptions and summarization
        self.transcript_buffer = []         # collects transcribed text chunks
        self.time_accumulator = 0.0         # seconds of audio accumulated
        self.summarize_interval_sec = 60    # call LLM every 60 seconds of audio
        self.context_summary = file_type    # initial context
        self.minute_count = 0

    def start(self):
        """
        Start capturing audio from the microphone in a background thread.
        Then, in the main thread, process audio chunks and call the LLM every 60 seconds.
        """
        self.stream = self.p.open(
            format=pyaudio.paFloat32,  # recording in float32
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        record_thread = threading.Thread(target=self._record_loop, daemon=True)
        record_thread.start()

        print("Recording from microphone... Press Ctrl+C to stop.\n")
        try:
            self._process_audio_chunks()
        except KeyboardInterrupt:
            print("KeyboardInterrupt received. Stopping...")
            self.stop()
        record_thread.join()

    def stop(self):
        """
        Signal the recording to stop and clean up the PyAudio stream.
        """
        self.stop_event.set()
        time.sleep(1)
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

    def _record_loop(self):
        """
        Continuously read from the microphone in small buffers.
        Once enough data for `chunk_sec` seconds is collected, push that chunk into the queue.
        """
        frames = []

        # For paFloat32, each sample is 4 bytes.
        total_bytes_needed = self.sample_rate * 4 * self.chunk_sec
        print(f"Capturing ~{self.chunk_sec}s lumps from microphone...")

        while not self.stop_event.is_set():
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            frames.append(data)
            if sum(len(x) for x in frames) >= total_bytes_needed:
                chunk_data = b''.join(frames)
                frames = []
                self.audio_queue.put(chunk_data)
        print("[_record_loop] Stopped reading from microphone.")

    def _process_audio_chunks(self):
        """
        Main processing loop: for each audio chunk received from the queue,
        send it to the STT microservice, accumulate the transcription,
        and once 60 seconds of text is reached, call the LLM microservice to process it.
        """
        while not self.stop_event.is_set():
            try:
                chunk_data = self.audio_queue.get(timeout=1.0)
            except:
                continue

            # Transcribe the current chunk using the STT microservice.
            transcription = transcribe_chunk_via_grpc(
                audio_chunk=chunk_data,
                stt_address=self.stt_address
            )
            print(f"[Microphone chunk] {transcription}")

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
                    # Expecting a JSON response with at least a "chunk_summary" field.
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
