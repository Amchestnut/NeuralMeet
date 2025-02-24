import grpc
from client.proto_repo import audio_pb2, audio_pb2_grpc


def transcribe_chunk_via_grpc(audio_chunk: bytes,
                              stt_address: str = "localhost:50051") -> str:
    """
    Sends a single chunk of audio to the STT microservice via gRPC.
    Returns the transcription text, or an empty string on error.
    """
    try:
        # Create the channel and stub
        with grpc.insecure_channel(stt_address) as channel:
            stub = audio_pb2_grpc.AudioStreamStub(channel)

            def request_generator():
                # 1) yield the actual audio chunk
                yield audio_pb2.AudioChunk(audio_data=audio_chunk)
                # 2) yield an empty chunk to signal end of stream
                yield audio_pb2.AudioChunk(audio_data=b'')

            # The STT service returns a stream of STTResponse.
            # Typically we'll get just one STTResponse with the final transcription.
            response_iterator = stub.StreamAudio(request_generator())

            for response in response_iterator:
                return response.transcription

    except Exception as e:
        print(f"[transcribe_chunk_via_grpc] Error: {e}")

    return ""
