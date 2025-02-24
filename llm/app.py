from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from llm.llm_processing import process_text

app = FastAPI()

class LLMRequest(BaseModel):
    text: str
    user_options: dict  # {"file_type": "meeting"}, or "lecture", or "call"
    rolling_context: str = ""

@app.post("/process_text")
async def process_text_endpoint(req: LLMRequest):
    chunk_summary, updated_context = process_text(req.text, req.user_options, req.rolling_context)
    return {"chunk_summary": chunk_summary, "updated_context": updated_context}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
