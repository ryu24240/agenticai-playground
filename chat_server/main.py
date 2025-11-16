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
    
class ChatResponse(BaseModel):
    reply: str

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://semantic_kernel:8100")

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
            response = await client.post(f"{ORCHESTRATOR_URL}/orchestrate", json=json_payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            # reply = data.get("reply", "(no reply)")
        return ChatResponse(reply=data.get("reply", "(no reply)"))
        # echo_reply = "You Said:" + req.messages[-1].content
        # return ChatResponse(reply=echo_reply)