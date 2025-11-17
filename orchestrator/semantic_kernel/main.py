import os

from pydantic import BaseModel
from typing import List

from openai import AsyncOpenAI
from fastapi import FastAPI, HTTPException

from semantic_kernel.kernel import Kernel, ChatHistory
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion, OllamaChatPromptExecutionSettings
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread

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
    
LLAMA_ENDPOINT = os.getenv("LLM_URL", "http://localhost:11434")
QWEN_ENDPOINT = os.getenv("MODEL_ENDPOINT")
QWEN_MODEL_NAME = os.getenv("MODEL_NAME")
OPENAI_API_KEY = "this is dummy"

async_openai = AsyncOpenAI(base_url=QWEN_ENDPOINT, api_key=OPENAI_API_KEY)

app = FastAPI(title="Semantic Kernel Orchestrator")
kernel = Kernel()

llama_service = OllamaChatCompletion(
        ai_model_id="llama3.1:latest", host=LLAMA_ENDPOINT
    )
qwen_service = OpenAIChatCompletion(
        ai_model_id=QWEN_MODEL_NAME,
        api_key=OPENAI_API_KEY,
        async_client=async_openai,
    )

llama_agent = ChatCompletionAgent(
    service=llama_service,
    name="llama-agent",
    instructions="You are a helpful assistant using local llama.",
)

qwen_agent = ChatCompletionAgent(
    service=qwen_service,
    name="qwen-agent",
    instructions="You are a helpful assistant using Qwen model.",
)

# kernel.add_service(llama_service)
# kernel.add_service(qwen_service)

# llama_execution_settings = OllamaChatPromptExecutionSettings()
# qwen_execution_settings = OpenAIChatPromptExecutionSettings()

# chat_history = ChatHistory()
thread = ChatHistoryAgentThread()

@app.post("/orchestrate")
async def orchestrate(req: ChatRequest) -> ChatResponse:
    if not req.messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    
    last_user_message = req.messages[-1].content
    model = req.model or "llama"

    if model == "llama":
        agent = llama_agent
    elif model == "qwen":
        agent = qwen_agent
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")
    
    res = await agent.get_response(
        messages=last_user_message,
        thread=thread,
    )

    return ChatResponse(reply=str(res))
    

    # else: 
    #     chat_history.add_user_message(req.messages[-1].content)
    #     res = await llama_service.get_chat_message_content(
    #         chat_history = chat_history,
    #         settings = execution_settings,
    #     )
        
    #     return ChatResponse(reply=str(res))
