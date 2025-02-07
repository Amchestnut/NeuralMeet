import time
# import Transcriber
from Summarizer import Summarizer
from Transcriber import Transcriber

# --- Configuration ---
# MEDIA_FILE = "resources/2024-07-17 Council Meeting.mp3"       # Shorter meeting audio file
# MEDIA_FILE = "resources/2024-08-20 Planning Meeting.mp3"      # Big

# MEDIA_FILE = "resources/ASP.mp4"                              # Lecture

MEDIA_FILE = "resources/phone_call_example.mp3"                 # Phone call


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


def run_realtime_processing():
    """Simulate real-time transcript processing for 5 minutes."""
    summarizer = Summarizer()
    meeting_duration_seconds = 5 * 60  # Simulated real-time meeting
    meeting_end_time = time.time() + meeting_duration_seconds

    while time.time() < meeting_end_time:
        # Simulating real-time incoming transcript text
        new_text = "This is a simulated transcript text. Discussing the new policy changes."
        summarizer.update_chunk(new_text)
        time.sleep(5)  # Simulated delay

    # Process any remaining transcript text
    final_summary = summarizer.get_final_summary()
    print("\nFinal Meeting Summary:\n", final_summary)


if __name__ == "__main__":
    print("Choose processing mode:")
    print("1. Process a full audio/video file")
    print("2. Run real-time transcript processing")
    mode_choice = input("Enter 1 or 2: ").strip()

    if mode_choice == "1":
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

        # You can also ask the user to enter the path to the file if you want:
        # media_file = input("Enter the path to the audio/video file: ").strip()
        # For now, we'll just use the default MEDIA_FILE
        run_audio_file_processing(MEDIA_FILE, chosen_file_type)

    elif mode_choice == "2":
        run_realtime_processing()
    else:
        print("Invalid choice. Exiting.")
