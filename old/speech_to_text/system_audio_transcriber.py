import sys
import os
import wave
import whisper
from queue import Queue
import threading


class SystemAudioTranscriber:
    """
    Reads raw PCM audio data from sys.stdin (piped in from FFmpeg).

    1) The user runs something like:
       ffmpeg -f dshow -i audio="Stereo Mix (Realtek)" -ar 16000 -ac 1 -f s16le - | wsl python -u wsl_system_audio_realtime.py
    2) This class reads from sys.stdin in chunks of N seconds.
    3) Each chunk is transcribed with Whisper.

    We store chunked frames in a queue so that a separate processor thread can handle them.
    """

    def __init__(self, model_size="medium", chunk_sec=5, sample_rate=16000):
        self.model = whisper.load_model(model_size)
        self.chunk_sec = chunk_sec
        self.sample_rate = sample_rate

        # 16-bit (2 bytes), single channel => bytes per second = sample_rate * 2
        # We'll read chunk_sec * sample_rate * 2 bytes for each chunk
        self.bytes_per_second = sample_rate * 2
        self.chunk_bytes = self.bytes_per_second * self.chunk_sec

        self.stop_event = threading.Event()
        self.audio_queue = Queue()

        self.thread = threading.Thread(target=self._capture_audio, daemon=True)

    def start_recording(self):
        """
        Starts a background thread that continuously reads raw PCM from stdin
        and puts chunked frames into self.audio_queue.
        """
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join()

    def _capture_audio(self):
        """
        Continuously read chunk_bytes from sys.stdin.
        Once we read chunk_bytes, we pass that frames chunk into the queue.
        """
        print("FFmpegSystemAudioTranscriber: start reading from stdin.")
        while not self.stop_event.is_set():
            # Attempt to read chunk_bytes from stdin
            data = sys.stdin.buffer.read(self.chunk_bytes)
            if not data or len(data) < self.chunk_bytes:
                # Reached EOF or stream ended
                break
            # Put the frames in the queue
            self.audio_queue.put(data)
        print("FFmpegSystemAudioTranscriber: stopped reading from stdin.")


    def transcribe_chunk(self, frames: bytes) -> str:
        """
        frames is raw PCM data (s16le).
        We'll write it to a temp WAV file, then call Whisper transcribe.
        """
        temp_wav = "temp_system_audio_chunk.wav"
        with wave.open(temp_wav, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(frames)

        result = self.model.transcribe(temp_wav)
        return result["text"].strip()
