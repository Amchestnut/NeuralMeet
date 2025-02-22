import os
from processing.audio_file_processor import AudioFileProcessor
from processing.realtime_processor import RealtimeProcessor


MEDIA_FILE = "resources/2024-07-17 Council Meeting.mp3"       # Shorter meeting audio file
# MEDIA_FILE = "resources/2024-08-20 Planning Meeting.mp3"      # Big
# MEDIA_FILE = "resources/ASP.mp4"                              # Lecture
# MEDIA_FILE = "resources/phone_call_example.mp3"               # Phone call
# MEDIA_FILE = "resources/machine_learning_lecture.mp4"
# MEDIA_FILE = "resources/ui.mp4"

def main():
    print("Choose processing mode:")
    print("1. Process a full audio/video file")
    print("2. Run real-time transcript processing")
    mode_choice = input("Enter 1 or 2: ").strip()

    print("What type of file is this?")
    print("1) Meeting")
    print("2) Lecture")
    print("3) Phone Call")
    file_type_choice = input("Enter 1, 2, or 3: ").strip()

    if file_type_choice == "1":
        chosen_file_type = "meeting"
    elif file_type_choice == "2":
        chosen_file_type = "lecture"
    elif file_type_choice == "3":
        chosen_file_type = "call"
    else:
        print("Invalid file type choice. Exiting.")
        return

    if mode_choice == "1":
        # Audio file processing
        processor = AudioFileProcessor(model_size="medium")
        processor.process_file(MEDIA_FILE, chosen_file_type)
    elif mode_choice == "2":
        # Real-time processing
        rt_processor = RealtimeProcessor(
            summarize_interval_sec=30,
            chunk_sec=5,
            model_size="medium",
            file_type=chosen_file_type,
            # output_file="realtime_summary.txt"
            output_file="/mnt/c/Users/Windows11/Desktop/A/AquamarineML/NeuralMeet/Realtime_processing.md"
        )
        rt_processor.start()
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    # Ensuring 'output' folder exists
    if not os.path.exists("output"):
        os.makedirs("output")

    main()
