# stt/app.py
import asyncio
from concurrent import futures
import grpc
import uvicorn
from fastapi import FastAPI

from stt.classes.STT import STT
from stt.classes.AudioStreamServicer import AudioStreamServicer
from stt.proto_repo import audio_pb2_grpc

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "STT Service is running."}

def create_grpc_server():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    stt_instance = STT()
    audio_pb2_grpc.add_AudioStreamServicer_to_server(
        AudioStreamServicer(stt_function=stt_instance.transcribe), server)

    server.add_insecure_port('[::]:50051')
    return server

async def start_grpc_server():
    grpc_server = create_grpc_server()
    grpc_server.start()
    print("gRPC STT server started on port 50051")
    grpc_server.wait_for_termination()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(start_grpc_server())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
