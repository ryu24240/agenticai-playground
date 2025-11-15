from pydantic import BaseModel
from typing import List

from fastapi import FastAPI

class Message(BaseModel):
    role: str
    content: str
    
class ChatRequest(BaseModel):
    session_id: str
    messages: List[Message]
    
class ChatResponse(BaseModel):
    reply: str


app = FastAPI(title="Chat Server")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest) -> ChatResponse:
    echo_reply = "You Said:" + req.messages[-1].content
    return ChatResponse(reply=echo_reply)