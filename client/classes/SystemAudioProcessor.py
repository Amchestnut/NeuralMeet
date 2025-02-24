import soundcard as sc
import threading
import time
import numpy as np
from queue import Queue

from .stt_client import transcribe_chunk_via_grpc

class SystemAudioProcessor:
    """
    Captures system playback audio (via WASAPI loopback) in 5-second chunks
    and sends them to the STT service.
    """
    def __init__(self,
                 stt_address="localhost:50051",
                 chunk_sec=5,
                 sample_rate=16000):
        self.stt_address = stt_address
        self.chunk_sec = chunk_sec
        self.sample_rate = sample_rate
        self.stop_event = threading.Event()
        self.audio_queue = Queue()

    def start(self):
        record_thread = threading.Thread(target=self._record_loop, daemon=True)
        record_thread.start()

        print("Recording system audio... Press Ctrl+C to stop.\n")
        self._process_audio_chunks()

    def stop(self):
        self.stop_event.set()

    def _record_loop(self):
        # Use the default loopback microphone for capturing system audio.
        loopback_mic = sc.default_microphone()
        print(f"Capturing system audio from: {loopback_mic.name}")
        with loopback_mic.recorder(samplerate=self.sample_rate,
                                   channels=1,
                                   blocksize=1024) as recorder:
            while not self.stop_event.is_set():
                # Record chunk_sec seconds of audio.
                data = recorder.record(numframes=self.sample_rate * self.chunk_sec)
                # data is a NumPy array (float32); convert it to bytes.
                chunk_data = data.tobytes()
                self.audio_queue.put(chunk_data)
        print("[_record_loop] Stopped recording system audio.")

    def _process_audio_chunks(self):
        while not self.stop_event.is_set():
            try:
                chunk_data = self.audio_queue.get(timeout=1.0)
            except Exception:
                continue
            # Send chunk to STT microservice.
            transcription = transcribe_chunk_via_grpc(
                audio_chunk=chunk_data,
                stt_address=self.stt_address
            )
            print(f"[System Audio chunk] {transcription}")
