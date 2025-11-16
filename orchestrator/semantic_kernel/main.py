import os

from pydantic import BaseModel
from typing import List

from fastapi import FastAPI, HTTPException
from semantic_kernel.kernel import Kernel, ChatHistory
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion, OllamaChatPromptExecutionSettings

class Message(BaseModel):
    role: str
    content: str
    
class ChatRequest(BaseModel):
    session_id: str
    messages: List[Message]
    
class ChatResponse(BaseModel):
    reply: str
    
model_endpoint = os.getenv("LLM_URL", "http://localhost:11434")

app = FastAPI(title="Semantic Kernel Orchestrator")
kernel = Kernel()

chat_completion = OllamaChatCompletion(ai_model_id="llama3.1:latest", host=model_endpoint)
kernel.add_service(chat_completion)
execution_settings = OllamaChatPromptExecutionSettings()
chat_history = ChatHistory()

@app.post("/orchestrate")
async def orchestrate(req: ChatRequest) -> ChatResponse:
    if not req.messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    else: 
        chat_history.add_user_message(req.messages[-1].content)
        res = await chat_completion.get_chat_message_content(
            chat_history = chat_history,
            settings = execution_settings,
        )
        
        return ChatResponse(reply=str(res))
