import whisper

# Load the model (already downloaded)
model = whisper.load_model("base")


# Transcribe an audio file of a meeting (.wav, .mp3 or .mp4 whatever)
result = model.transcribe("resources/2024-07-17 Council Meeting.mp3")

# Print the transcribed text
print("Transcribed Text:\n", result["text"])
