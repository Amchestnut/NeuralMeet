import whisper
import pyaudio
import wave


def record_audio(filename="recorded_audio.wav", duration=5, sample_rate=16000):
    """
    Records audio from the microphone for a given duration.
    Saves it as a WAV file.
    """
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1  # Mono audio
    RATE = sample_rate

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print(f"Recording for {duration} seconds...")
    frames = []

    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()


def transcribe_audio(file_path="recorded_audio.wav"):
    """
    Uses Whisper to transcribe an audio file.
    """
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    print("\nTranscribed Text:\n", result["text"])


# Step 1: Record a 5-second audio
record_audio()

# Step 2: Transcribe the recorded audio
transcribe_audio()
