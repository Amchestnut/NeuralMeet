import whisper

class Whisper:
    def __init__(self, model_size="medium"):
        """
        Initialize the Whisper model.
        :param model_size: e.g., 'small', 'medium', 'large', etc.
        """
        self.model = whisper.load_model(model_size)

    def transcribe(self, file_path):
        """
        Transcribe a given audio (or video) file and return the full text.
        :param file_path: Path to the audio file.
        """
        result = self.model.transcribe(file_path, fp16=False)
        return result["text"]