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
LANGGRAPH_URL = os.getenv("LANGGRAPH_URL", "http://langgraph:8300")

app = FastAPI(title="Chat Server")

## TODO: 会話セッション管理（過去の会話履歴取り出し等）・バリデーション・ストリーミング対応追加

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
async def chat(req: ChatRequest) -> ChatResponse:
    if not req.messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    else:
        json_payload = req.model_dump()
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=180.0)) as client:
            if req.orchestrator == "Semantic Kernel":
                response = await client.post(f"{SEMANTIC_KERNEL_URL}/orchestrate", json=json_payload)
                response.raise_for_status()
                data = response.json()

                return ChatResponse(reply=data.get("reply", "(no reply)"))
            
            elif req.orchestrator == "LangGraph":
                response = await client.post(f"{LANGGRAPH_URL}/orchestrate", json=json_payload)
                response.raise_for_status()
                data = response.json()

                return ChatResponse(reply=data.get("reply", "(no reply)"))
        # echo_reply = "You Said:" + req.messages[-1].content
        # return ChatResponse(reply=echo_reply)