import os
import httpx

from pydantic import BaseModel
from typing import List

from fastapi import FastAPI, HTTPException

class Message(BaseModel):
    role: str
    content: str
    
class ChatRequest(BaseModel):
    session_id: str
    messages: List[Message]
    orchestrator: str | None = None
    model: str | None = None
    
class ChatResponse(BaseModel):
    reply: str

SEMANTIC_KERNEL_URL = os.getenv("SEMANTIC_KERNEL_URL", "http://semantic_kernel:8100")

app = FastAPI(title="Chat Server")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
async def chat(req: ChatRequest) -> ChatResponse:
    if not req.messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    else:
        json_payload = req.model_dump()
        async with httpx.AsyncClient() as client:
            if req.orchestrator == "Semantic Kernel":
                response = await client.post(f"{SEMANTIC_KERNEL_URL}/orchestrate", json=json_payload, timeout=60)
                response.raise_for_status()
                data = response.json()

                return ChatResponse(reply=data.get("reply", "(no reply)"))
        # echo_reply = "You Said:" + req.messages[-1].content
        # return ChatResponse(reply=echo_reply)