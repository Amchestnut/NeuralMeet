import pyaudio
import wave
import time
import whisper
import os
from Summarizer import Summarizer
import sys


class RealtimeTranscriber:
    """
    Continuously captures audio from a microphone in ~5-second chunks,
    transcribes each chunk with Whisper, and every 1 minute:
      - Summarizes that minute's text using Summarizer's chunk-level prompt
      - Maintains a short rolling context summary
      - Appends the newly summarized text to a TXT file
    """

    def __init__(
        self,
        output_file="realtime_summary.txt",
        model_size="base",
        chunk_sec=5,
        summarize_interval_sec=60,
        file_type="meeting"
    ):
        """
        :param output_file: The TXT file path to append chunk summaries.
        :param model_size: Whisper model size (e.g. 'base', 'small', etc.)
        :param chunk_sec: Each chunk's duration in seconds before transcription
        :param summarize_interval_sec: The interval (in seconds) at which we do a chunk summary
        :param file_type: 'meeting', 'lecture', or 'call' for Summarizer context
        """
        self.is_recording = True
        self.output_file = output_file
        self.chunk_sec = chunk_sec
        self.summarize_interval_sec = summarize_interval_sec

        # Load the Whisper model once
        print(f"Loading Whisper model '{model_size}' ...")
        self.model = whisper.load_model(model_size)

        # Create Summarizer
        self.summarizer = Summarizer(file_type=file_type)

        # Internal buffers and counters
        self._transcript_buffer = []    # stores partial transcripts for the current interval
        self._time_accumulator = 0.0    # how many seconds since the last summarization
        self._context_summary = ""      # rolling context between each minute chunk
        self._minute_count = 0          # keeps track of how many 1-minute chunks have passed

        if os.path.exists(self.output_file):
            os.remove(self.output_file)


    def _record_chunk(self, stream, chunk_size, rate):
        """
        Record self.chunk_sec seconds of audio from the stream.
        Returns the raw frames.
        """
        all_chunks = []
        num_of_chunks = int(rate / chunk_size * self.chunk_sec)

        for _ in range(num_of_chunks):
            # We read 1024 frames from the microphone at once. Also if the mic buffer gets too full (overflow), we just drop some data
            try:
                one_chunk = stream.read(chunk_size, exception_on_overflow=False)
            except KeyboardInterrupt:       # We want to be able to stop the processing
                print("\nStopping real-time transcription...")
                raise
            all_chunks.append(one_chunk)

        # The data we want to return needs to be in BYTES, so 'b' tells python to join the binary chunks as binary data
        return b''.join(all_chunks)


    def _transcribe_chunk(self, frames, channels=1, rate=16000):
        """
        Write PCM frames to a small WAV file, then transcribe with Whisper.
        Returns recognized text for that chunk.
        """
        temp_wav = "temp_chunk.wav"

        wf = wave.open(temp_wav, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # pyaudio.paInt16 = 2 bytes
        wf.setframerate(rate)
        wf.writeframes(frames)
        wf.close()

        result = self.model.transcribe(temp_wav)
        text = result["text"].strip()

        return text


    def _summarize_minute(self, minute_text):
        """
        Use Summarizer's chunk-level approach to summarize the minute_text, carrying a rolling context_summary forward.
        Summarizer has an internal method `_process_chunk` that returns (chunk_summary, updated_context)
        We pass in our `minute_text` plus the current `_context_summary`.
        """

        chunk_summary, updated_context = self.summarizer._process_chunk(
            chunk_text=minute_text,
            context_summary=self._context_summary if self._context_summary else "No previous context."
        )
        return chunk_summary, updated_context


    def run(self):
        """
        Main loop:
          - Capture audio in 5s chunks
          - Transcribe each chunk
          - Every 1 min, summarize that minute's transcript with rolling context
          - Append the summarized chunk to a TXT file
        """
        chunk_size = 1024   # 1024 frames per 1 chunk
        rate = 16000        # 16000 frames per second (higher rate = higher quality = bigger size)      (cd quality is usually around 45000)
        channels = 1

        # So in 1 second we need: 16000/1024 ≈ 15.625 chunks
        # 1 chunk = 5 sec, For 5 seconds we need: 15.625 * 5 = 78.125 chunks (rounded to 78)

        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,         # 16-bit audio (CD quality)
                        channels=channels,              # 1: Mono audio
                        rate=rate,                      # 16000 samples per sec
                        input=True,                     # We want to RECORD, not PLAT
                        frames_per_buffer=chunk_size)   # Process 1024 frame at a time

        print("=== Real-time Transcription & Summaries ===")
        print("Capturing microphone audio, press Ctrl+C to stop.\n")

        try:
            while self.is_recording:
                # 1) Capture ~5 seconds of audio
                frames = self._record_chunk(stream, chunk_size, rate)

                # 2) Convert speech to text
                partial_text = self._transcribe_chunk(frames, channels, rate)
                print(f"[Partial text]: {partial_text}")

                # 3) Accumulate partial_text, adding to our 1-minute buffer (future queue)
                self._transcript_buffer.append(partial_text)
                self._time_accumulator += self.chunk_sec

                # 4) Check if we've hit 1 minute
                if self._time_accumulator >= self.summarize_interval_sec:
                    self._minute_count += 1

                    # Combine all partial transcripts for this past minute
                    minute_text = "\n".join(self._transcript_buffer)

                    # Summarize the minute text (rolling context)
                    chunk_part, updated_context = self._summarize_minute(minute_text)

                    # Print or log the chunk text part
                    print(f"\n--- Minute {self._minute_count} Chunk text part summary: ---")
                    print(chunk_part)
                    print("--- End Summary ---\n")

                    # Append summary to output file
                    with open(self.output_file, "a", encoding="utf-8") as f:
                        f.write(f"\n## Minute {self._minute_count}\n")
                        f.write(f"**Context before chunk**: {self._context_summary}\n\n")
                        f.write(f"**New Summary**:\n{chunk_part}\n\n")

                    # Update rolling context, reset buffers
                    self._context_summary = updated_context
                    self._transcript_buffer = []
                    self._time_accumulator = 0.0

        except KeyboardInterrupt:
            self.is_recording = False
        except Exception as e:
            print(f"Error: {e}")
            self.is_recording = False
        finally:
            self.is_recording = False
            # Cleanup audio
            stream.stop_stream()
            stream.close()
            p.terminate()

            # Optionally handle leftover partial text
            if self._transcript_buffer:
                leftover_text = "\n".join(self._transcript_buffer)
                # We won't do a final summary for now, probably never:
                # chunk_summary, updated_context = self._summarize_minute(leftover_text)

                print("\nThere were some leftover transcript lines not included in a 1-minute summary:")
                print(leftover_text)

            print(f"All done. Summaries appended to: {self.output_file}")


if __name__ == "__main__":
    rt = RealtimeTranscriber(
        output_file="realtime_summary.txt",
        model_size="base",
        chunk_sec=5,
        summarize_interval_sec=60,
        file_type="meeting"
    )

    # Check mic audio
    # p = pyaudio.PyAudio()
    # for i in range(p.get_device_count()):
    #     devinfo = p.get_device_info_by_index(i)
    #     print(i, devinfo.get('name'), devinfo.get('maxInputChannels'))

    rt.run()
