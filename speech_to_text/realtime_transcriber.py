import pyaudio
import wave
import time
import whisper
import os
import sys
import threading
from queue import Queue

class RealtimeTranscriber:
    """
    Continuously captures audio from the microphone and enqueues audio chunks.
    A separate processing thread can then transcribe these chunks in parallel.
    This class ONLY handles audio capture and raw transcription, not summarization.
    """

    def __init__(
        self,
        model_size="medium",
        chunk_sec=5,
        rate=16000,
        channels=1
    ):
        self.stop_event = threading.Event()
        self.audio_queue = Queue()
        self.chunk_sec = chunk_sec
        self.rate = rate
        self.channels = channels

        print(f"Loading Whisper model '{model_size}' ...")
        self.model = whisper.load_model(model_size)

    def _record_chunk(self, stream, chunk_size):
        """
        Record `self.chunk_sec` seconds of audio (using multiple small reads) and return raw frames.
        """
        frames = []
        num_of_reads = int(self.rate / chunk_size * self.chunk_sec)
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
        Producer thread: continuously record audio from the microphone and put each chunk on the queue.
        """
        chunk_size = 1024
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=self.channels,
                        rate=self.rate,
                        input=True,
                        frames_per_buffer=chunk_size)
        print("Recording thread started.")
        while not self.stop_event.is_set():
            frames = self._record_chunk(stream, chunk_size)
            if frames:
                self.audio_queue.put(frames)
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Recording thread stopped.")

    def transcribe_chunk(self, frames):
        """
        Writes the recorded frames to a temporary WAV file and transcribes them using Whisper.
        """
        temp_wav = "temp_chunk.wav"
        wf = wave.open(temp_wav, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(2)  # pyaudio.paInt16 = 2 bytes
        wf.setframerate(self.rate)
        wf.writeframes(frames)
        wf.close()
        result = self.model.transcribe(temp_wav)
        text = result["text"].strip()
        return text

    def stop(self):
        """
        Signal the recording to stop (e.g., from outside a loop).
        """
        self.stop_event.set()
