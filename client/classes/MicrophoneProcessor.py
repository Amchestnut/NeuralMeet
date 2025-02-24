import pyaudio
import threading
import time
from queue import Queue

from .stt_client import transcribe_chunk_via_grpc

class MicrophoneProcessor:
    def __init__(self,
                 stt_address="localhost:50051",
                 chunk_sec=5,
                 sample_rate=16000,
                 channels=1):
        self.stt_address = stt_address
        self.chunk_sec = chunk_sec
        self.sample_rate = sample_rate
        self.channels = channels

        self.stop_event = threading.Event()
        self.audio_queue = Queue()

        # PyAudio settings
        self.p = pyaudio.PyAudio()
        self.chunk_size = 1024  # frames per buffer
        self.stream = None

    def start(self):
        """
        Start capturing audio from the microphone in a background thread.
        Then in the main thread, read from audio_queue in 5s lumps and transcribe them.
        """
        # Start the PyAudio stream
        self.stream = self.p.open(
            # format=pyaudio.paInt16,
            format=pyaudio.paFloat32,  # record in float32 instead of paInt16
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        record_thread = threading.Thread(target=self._record_loop, daemon=True)
        record_thread.start()

        print("Recording from microphone... Press Ctrl+C to stop.\n")
        self._process_audio_chunks()

    def stop(self):
        """
        Signal the background thread to stop reading from mic.
        """
        self.stop_event.set()
        time.sleep(1)
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

    def _record_loop(self):
        """
        Continuously read from the microphone in small buffers,
        collect up to chunk_sec, then push that chunk into the queue.
        """
        frames = []
        total_bytes_needed = self.sample_rate * 4 * self.chunk_sec  # 5 bytes per sample
        print(f"Capturing ~{self.chunk_sec}s lumps from microphone...")

        while not self.stop_event.is_set():
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            frames.append(data)

            # Check if we have enough bytes for 5 seconds
            if sum(len(x) for x in frames) >= total_bytes_needed:
                # Combine into one chunk
                chunk_data = b''.join(frames)
                frames = []
                self.audio_queue.put(chunk_data)

        print("[_record_loop] Stopped reading from microphone.")

    def _process_audio_chunks(self):
        """
        Main loop: block on self.audio_queue,
        for each 5-second chunk, call STT.
        """
        while not self.stop_event.is_set():
            try:
                chunk_data = self.audio_queue.get(timeout=1.0)
            except:
                continue  # queue empty, just loop

            # Send chunk to STT microservice
            transcription = transcribe_chunk_via_grpc(
                audio_chunk=chunk_data,
                stt_address=self.stt_address
            )
            print(f"[Microphone chunk] {transcription}")
