import time
# import Transcriber
from Summarizer import Summarizer
from Transcriber import Transcriber

# --- Configuration ---
# MEETING_FILE = "resources/2024-07-17 Council Meeting.mp3"  # Meeting audio file
# MEETING_FILE = "resources/ASP.mp4"
MEETING_FILE = "resources/2024-08-20 Planning Meeting.mp3"

def run_full_meeting_processing():
    """Process a full meeting audio file, transcribe it, and summarize."""
    transcriber = Transcriber()
    summarizer = Summarizer()

    print("Transcribing full meeting...")
    transcript = transcriber.transcribe_audio(MEETING_FILE)

    print("Summarizing full meeting...")
    final_summary = summarizer.summarize(transcript)

    print("\nFinal Meeting Summary:\n", final_summary)


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
    print("1. Process a full meeting file")
    print("2. Run real-time transcript processing")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        run_full_meeting_processing()
    elif choice == "2":
        run_realtime_processing()
    else:
        print("Invalid choice. Exiting.")
