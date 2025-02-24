import os
from classes.FileProcessor import FileProcessor
from classes.MicrophoneProcessor import MicrophoneProcessor
from classes.SystemAudioProcessor import SystemAudioProcessor

# MEDIA_FILE = "resources/2024-07-17 Council Meeting.mp3"       # Shorter meeting audio file
# MEDIA_FILE = "resources/2024-08-20 Planning Meeting.mp3"      # Big
# MEDIA_FILE = "resources/ASP.mp4"                              # Lecture
MEDIA_FILE = "resources/phone_call_example.mp3"               # Phone call
# MEDIA_FILE = "resources/machine_learning_lecture.mp4"
# MEDIA_FILE = "resources/ui.mp4"

def main():
    # STT microservice location:
    stt_host = "localhost"
    stt_port = 50051
    stt_address = f"{stt_host}:{stt_port}"

    print("Choose processing mode:")
    print("1. Process a full audio/video file")
    print("2. Run real-time microphone processing")
    print("3. Run real-time system audio processing")
    mode_choice = input("Enter 1, 2, or 3: ").strip()

    print("\nWhat type of file/audio is this?")
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
        # Process a full audio/video file
        processor = FileProcessor(
            stt_address=stt_address,
            chunk_sec=5,
            sample_rate=16000,
            llm_endpoint="http://localhost:8001/process_text",
            output_file="C:/Users/Windows11/Desktop/A/AquamarineML/NeuralMeet/Audio_file_processing.md"
        )
        processor.process_file(MEDIA_FILE, chosen_file_type)

    elif mode_choice == "2":
        # Microphone real-time
        mic_proc = MicrophoneProcessor(
            stt_address=stt_address,
            chunk_sec=10,
            sample_rate=16000,
            channels=1,
            llm_endpoint="http://localhost:8001/process_text",
            file_type=chosen_file_type,
            output_file="C:/Users/Windows11/Desktop/A/AquamarineML/NeuralMeet/Microphone_realtime_processing.md"
        )
        try:
            mic_proc.start()  # blocks until user Ctrl+C or smt
        except KeyboardInterrupt:
            pass
        finally:
            mic_proc.stop()

    elif mode_choice == "3":
        # System audio real-time
        sys_proc = SystemAudioProcessor(
            stt_address=stt_address,
            chunk_sec=10,
            sample_rate=16000,
            llm_endpoint="http://localhost:8001/process_text",
            file_type=chosen_file_type,
            output_file="C:/Users/Windows11/Desktop/A/AquamarineML/NeuralMeet/System_audio_realtime_processing.md"
        )
        try:
            sys_proc.start()  # blocks
        except KeyboardInterrupt:
            pass
        finally:
            sys_proc.stop()

    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    # Make sure any needed folders exist, etc.
    if not os.path.exists("resources"):
        os.makedirs("resources")

    main()
