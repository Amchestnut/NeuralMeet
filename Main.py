from Summarizer import Summarizer
from Transcriber import Transcriber
from RealtimeTranscriber import RealtimeTranscriber


# MEDIA_FILE = "resources/2024-07-17 Council Meeting.mp3"       # Shorter meeting audio file
# MEDIA_FILE = "resources/2024-08-20 Planning Meeting.mp3"      # Big

# MEDIA_FILE = "resources/ASP.mp4"                              # Lecture

MEDIA_FILE = "resources/phone_call_example.mp3"               # Phone call


def run_audio_file_processing(media_file, file_type):
    """
    Process a full audio/video file, transcribe it, and summarize.
    file_type can be 'meeting', 'lecture', or 'call' (phone call).
    """
    transcriber = Transcriber()

    # Pass file_type to Summarizer so it knows which prompts to use
    summarizer = Summarizer(file_type=file_type)

    print(f"Transcribing the {file_type} audio/video file...")
    transcript = transcriber.transcribe_audio(media_file)
    print(transcript)

    print(f"Summarizing the {file_type}...")
    final_summary = summarizer.summarize(transcript)

    print(f"\nFinal {file_type.capitalize()} Summary:\n", final_summary)

    # --- Write the final summary to a file ---
    output_file = f"output/final_{file_type}_summary.txt"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_summary)
        print(f"\nThe final {file_type} report has been saved to: {output_file}")
    except Exception as e:
        print("Error writing the final meeting report to file:", e)


def run_realtime_processing(chosen_file_type):
    """
    Process audio in realtime and summarize.
    """

    # Create and run the transcriber
    rt = RealtimeTranscriber(
        output_file="realtime_summary.txt",
        model_size="base",          # or small, medium...
        chunk_sec=5,                # capture audio in 5-second chunks
        summarize_interval_sec=60,  # summarize every 60 seconds
        file_type=chosen_file_type
    )
    rt.run()


if __name__ == "__main__":
    print("Choose processing mode:")
    print("1. Process a full audio/video file")
    print("2. Run real-time transcript processing")
    mode_choice = input("Enter 1 or 2: ").strip()

    # Prompt user for the file type
    print("What type of file is this?")
    print("1) Meeting")
    print("2) Lecture")
    print("3) Phone Call")
    file_type_choice = input("Enter 1, 2, or 3: ").strip()

    # Map user choice to a string
    if file_type_choice == "1":
        chosen_file_type = "meeting"
    elif file_type_choice == "2":
        chosen_file_type = "lecture"
    elif file_type_choice == "3":
        chosen_file_type = "call"
    else:
        print("Invalid file type choice. Exiting.")
        exit()


    if mode_choice == "1":
        run_audio_file_processing(MEDIA_FILE, chosen_file_type)
    elif mode_choice == "2":
        run_realtime_processing(chosen_file_type)
    else:
        print("Invalid choice. Exiting.")
