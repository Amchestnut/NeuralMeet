from stt.proto_repo import audio_pb2, audio_pb2_grpc
import numpy as np

class AudioStreamServicer(audio_pb2_grpc.AudioStreamServicer):
    def __init__(self, stt_function):
        """
        :param stt_function: a callable that receives a NumPy array and returns a transcription string.
        """
        self.stt_function = stt_function

    def StreamAudio(self, request_iterator, context):
        full_audio = None

        # Loop through the received chunks.
        for audio_chunk in request_iterator:
            data = audio_chunk.audio_data

            # An empty audio chunk indicates the end of the stream.
            if len(data) == 0:
                break

            # Convert the received bytes into a numpy array of float32.
            # (Assumes the client sends data as float32)
            audio_np = np.frombuffer(data, dtype=np.float32)
            # audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            if full_audio is None:
                full_audio = audio_np
            else:
                full_audio = np.concatenate((full_audio, audio_np))

        # If no audio was received, return an empty transcription.
        if full_audio is None:
            transcription = ""
        else:
            transcription = self.stt_function(full_audio)
        yield audio_pb2.STTResponse(transcription=transcription)
