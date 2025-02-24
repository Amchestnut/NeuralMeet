# import os
# import numpy as np
# from pydub import AudioSegment
# from pydub.utils import make_chunks
# from .stt_client import transcribe_chunk_via_grpc
#
# class FileProcessor:
#     def __init__(self,
#                  stt_address="localhost:50051",
#                  chunk_sec=5,
#                  sample_rate=16000):
#         """
#         :param stt_address: Host:port for STT microservice
#         :param chunk_sec: Chunk size in seconds
#         :param sample_rate: Desired sample rate to match STT service
#         """
#         self.stt_address = stt_address
#         self.chunk_sec = chunk_sec
#         self.sample_rate = sample_rate
#
#     def process_file(self, file_path: str, file_type: str):
#         """
#         1. Convert the file to the desired sample_rate, mono.
#         2. Break into ~5s chunks.
#         3. For each chunk, convert samples from int16 to normalized float32,
#            then call the STT microservice.
#         4. Print or accumulate the transcriptions.
#         """
#         print(f"Processing file: {file_path} as {file_type} ...")
#
#         # Load file via pydub
#         audio = AudioSegment.from_file(file_path)
#         # Convert to mono, target sample rate.
#         audio = audio.set_frame_rate(self.sample_rate).set_channels(1)
#
#         # Make 5-second chunks (pydub uses milliseconds)
#         chunk_length_ms = self.chunk_sec * 1000
#         chunks = make_chunks(audio, chunk_length_ms)
#
#         all_transcriptions = []
#
#         for i, chunk in enumerate(chunks):
#             print(f"Processing chunk #{i + 1}/{len(chunks)} ...")
#             # Convert chunk to a numpy array of int16 samples
#             samples = np.array(chunk.get_array_of_samples())
#             # Convert to float32 and normalize to [-1, 1] (16-bit PCM)
#             samples = samples.astype(np.float32) / 32768.0
#             # Convert to bytes for transmission
#             chunk_data = samples.tobytes()
#
#             # Send to STT microservice
#             transcription = transcribe_chunk_via_grpc(
#                 audio_chunk=chunk_data,
#                 stt_address=self.stt_address
#             )
#             print(f"Chunk #{i + 1} transcription: {transcription}")
#             all_transcriptions.append(transcription)
#
#         full_text = "\n".join(all_transcriptions)
#         print(f"\n--- Final Transcription ({file_type}) ---\n{full_text}\n")
#         # Optionally, write full_text to a file.


import os
import numpy as np
from pydub import AudioSegment
from pydub.utils import make_chunks
from .stt_client import transcribe_chunk_via_grpc
import requests

class FileProcessor:
    def __init__(self,
                 stt_address="localhost:50051",
                 chunk_sec=5,
                 sample_rate=16000,
                 llm_endpoint="http://localhost:8001/process_text",
                 output_file="/mnt/c/Users/Windows11/Desktop/A/AquamarineML/NeuralMeet/Realtime_processing.md"):
        """
        :param stt_address: Host:port for the STT microservice.
        :param chunk_sec: Duration of each chunk in seconds.
        :param sample_rate: Sample rate (in Hz) to convert the file to.
        :param llm_endpoint: Endpoint URL for the LLM microservice.
        :param output_file: Full path where the final summary will be written.
        """
        self.stt_address = stt_address
        self.chunk_sec = chunk_sec
        self.sample_rate = sample_rate
        self.llm_endpoint = llm_endpoint
        self.output_file = output_file

    def process_file(self, file_path: str, file_type: str):
        print(f"Processing file: {file_path} as {file_type} ...")

        # 1. Load the audio file using pydub.
        audio = AudioSegment.from_file(file_path)
        # Convert to mono and target sample rate.
        audio = audio.set_frame_rate(self.sample_rate).set_channels(1)

        # 2. Break the audio into chunks (chunk_sec seconds per chunk; pydub works in ms).
        chunk_length_ms = self.chunk_sec * 1000
        chunks = make_chunks(audio, chunk_length_ms)

        all_transcriptions = []

        # 3. Process each chunk:
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk #{i + 1}/{len(chunks)} ...")
            # Get the raw samples (as int16), convert to float32 and normalize to [-1, 1]
            samples = np.array(chunk.get_array_of_samples())
            samples = samples.astype(np.float32) / 32768.0
            # Convert the samples to bytes.
            chunk_data = samples.tobytes()

            # Call the STT microservice to transcribe this chunk.
            transcription = transcribe_chunk_via_grpc(
                audio_chunk=chunk_data,
                stt_address=self.stt_address
            )
            print(f"Chunk #{i + 1} transcription: {transcription}")
            all_transcriptions.append(transcription)

        # 4. Combine all chunk transcriptions into a full transcript.
        full_text = "\n".join(all_transcriptions)
        print(f"\n--- Full Transcript ({file_type}) ---\n{full_text}\n")

        # 5. Now send the full transcript to the LLM microservice for final processing.
        payload = {
            "text": full_text,
            "user_options": {"file_type": file_type},
            "rolling_context": file_type  # initial context is simply the file type
        }
        try:
            resp = requests.post(self.llm_endpoint, json=payload)
            resp.raise_for_status()
            data = resp.json()
            final_summary = data.get("chunk_summary", full_text)
        except Exception as e:
            print(f"Error calling LLM service: {e}")
            final_summary = full_text

        print(f"\n--- Final Summary ({file_type}) ---\n{final_summary}\n")

        # 6. Write the final summary to the output file.
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(final_summary)
