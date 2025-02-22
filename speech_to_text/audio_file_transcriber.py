import whisper

class AudioFileTranscriber:
    """
    A simple wrapper around the Whisper model for audio file (batch) transcription.
    """

    def __init__(self, model_size="medium"):
        """
        Initialize the Whisper model.
        :param model_size: e.g., 'small', 'medium', 'large', etc.
        """
        self.model = whisper.load_model(model_size)

    def transcribe_audio(self, file_path):
        """
        Transcribe a given audio (or video) file and return the full text.
        """
        result = self.model.transcribe(file_path)
        return result["text"]
