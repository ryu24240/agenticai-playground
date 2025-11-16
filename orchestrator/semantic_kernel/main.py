from pydantic import BaseModel
from typing import List

from fastapi import FastAPI
from semantic_kernel.kernel import Kernel

class Message(BaseModel):
    role: str
    content: str
    
class ChatRequest(BaseModel):
    session_id: str
    messages: List[Message]
    
class ChatResponse(BaseModel):
    reply: str

app = FastAPI(title="Semantic Kernel Orchestrator")
kernel = Kernel()

@app.post("/orchestrate")
def orchestrate(req: ChatRequest) -> ChatResponse:
    user_message = req.messages[-1].content
    res = mock_agent(user_message)
    return ChatResponse(reply=res)

def mock_agent(user_message: str):
    return f"You Said: {user_message}"