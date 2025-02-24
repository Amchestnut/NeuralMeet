from stt.classes.Whisper import Whisper

class STT:
    def __init__(self):
        self.whisper = Whisper()

    def transcribe(self, audio_data):
        # audio_data is a numpy array of audio samples.
        transcription = self.whisper.transcribe(audio_data)

        print("Transcription:", transcription)
        return transcription
