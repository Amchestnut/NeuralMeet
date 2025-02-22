import whisper

class Transcriber:
    def __init__(self, model_size="medium"):
        """Initialize the Whisper model."""
        self.model = whisper.load_model(model_size)

    def transcribe_audio(self, file_path):
        """Transcribe a given audio file and return the full text."""
        result = self.model.transcribe(file_path)
        return result["text"]
