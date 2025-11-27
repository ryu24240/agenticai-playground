from __future__ import annotations

import json
import os
import uuid
from contextlib import asynccontextmanager

import httpx
from pydantic import BaseModel
from typing import List, Annotated

from fastapi import FastAPI, HTTPException, Depends

from openai import AsyncOpenAI
from semantic_kernel.kernel import Kernel, ChatHistory
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion, OllamaChatPromptExecutionSettings
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior

from a2a.types import Message as A2AMessage, Part, Role
from a2a.types import TextPart as A2ATextPart


from a2aclient.client import A2AClient
from a2a_tools import A2ATools

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
QWEN_ENDPOINT = os.getenv("MODEL_ENDPOINT", "dummy-endpoint")
QWEN_MODEL_NAME = os.getenv("MODEL_NAME", "dummy-model")
OPENAI_API_KEY = "this is dummy"

REMOTE_AGENT_ADDRESSES: List[str] = [
    "http://faq_rag_agent:8200",
]

async_openai = AsyncOpenAI(base_url=QWEN_ENDPOINT, api_key=OPENAI_API_KEY)


kernel = Kernel()

llama_service = OllamaChatCompletion(
        ai_model_id="llama3.1:latest", host=LLAMA_ENDPOINT
    )
qwen_service = OpenAIChatCompletion(
        ai_model_id=QWEN_MODEL_NAME,
        api_key=OPENAI_API_KEY,
        async_client=async_openai,
    )

thread = ChatHistoryAgentThread()

# class A2AOrchestratorPlugin:
#     """
#     ChatCompletionAgent から呼び出される「ツール」としての A2A プラグイン。

#     - list_remote_agents()
#         -> 利用可能なリモートエージェント一覧を返す
#     - send_message_to_remote_agent(agent_name, message)
#         -> 指定したリモートエージェントに A2A でメッセージ送信
#     """

#     def __init__(self, a2a_client: A2AClient):
#         self._a2a_client = a2a_client

#     @kernel_function(
#         name="list_remote_agents",
#         description="List available remote A2A agents and their descriptions.",
#     )
#     def list_remote_agents(self) -> Annotated[str, "JSON array of {name, description}"]:
#         """
#         利用可能なリモートエージェントの一覧をJSON文字列で返す。
#         """
#         info = [
#             {"name": card.name, "description": card.description}
#             for card in self._a2a_client.cards
#         ]
#         return json.dumps(info, ensure_ascii=False)
    

#     @kernel_function(
#         name="send_message_to_remote_agent",
#         description=(
#             "Send a user message to a remote A2A agent by name and return the "
#             "agent's textual responses."
#         ),
#     )
#     async def send_message_to_remote_agent(
#         self,
#         agent_name: Annotated[str, "Name property of the remote AgentCard"],
#         message: Annotated[str, "User message to send to the remote agent."],
#     ) -> Annotated[str, "Concatenated text reply from the remote agent"]:
#         """
#         名前で指定されたリモートエージェントにメッセージを送り、
#         text パートを連結して返す。
#         """
#         client = self._a2a_client.clients.get(agent_name)
#         if client is None:
#             raise ValueError(f"Agent {agent_name!r} not found in A2A clients")

#         request_message = A2AMessage(
#             role=Role.user,
#             parts=[Part(root=A2ATextPart(text=message))],
#             message_id=str(uuid.uuid4()),
#         )

#         texts: list[str] = []
#         async for response in client.send_message(request_message):
#             for part in response.parts:
#                 if part.root.kind == "text":
#                     texts.append(part.root.text)

#         if not texts:
#             return "Remote agent returned no text parts."

#         return "\n".join(texts)




@asynccontextmanager
async def lifespan(app: FastAPI):
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=30.0))

    a2a_client = None
    a2a_plugin = None

    try: 
        a2a_client = await A2AClient.create(REMOTE_AGENT_ADDRESSES, http_client)
        a2a_plugin = A2ATools(a2a_client)
        print("[semantic_kernel] A2A client initialized. cards=",
              [c.name for c in a2a_client.cards])
    except Exception as e:
        print(f"[semantic_kernel] Failed to init A2A client: {e}")
        a2a_client = None
        a2a_plugin = None

    llama_agent = ChatCompletionAgent(
        service=llama_service,
        name="llama-agent",
        instructions="""
        You are an expert delegator agent inside a 'Agentic banking' AI system.

        - You can delegate user requests to remote agents over the A2A protocol.
        - Use the tool `list_remote_agents` to discover which remote agents are available.
        - For actionable banking tasks, use `send_message_to_remote_agent` to call a
        specific remote agent by its name.
        - Always base your answer on the actual tool results.
        - Always answer in Japanese.
        - When you delegate to a remote agent, clearly mention which agent you used in
        your final answer (e.g. 'Using Agentic Bank FAQ Agent: ...').
        - If the request is simple chit-chat or does not require any remote action,
        you may answer directly on your own.
        """,
        plugins=[a2a_plugin] if a2a_plugin is not None else [],
        function_choice_behavior=FunctionChoiceBehavior.Auto(),
    )

    qwen_agent = ChatCompletionAgent(
        service=qwen_service,
        name="qwen-agent",
        instructions="""
        You are an expert delegator agent inside a 'Agentic banking' AI system.

        - You can delegate user requests to remote agents over the A2A protocol.
        - Use the tool `list_remote_agents` to discover which remote agents are available.
        - For actionable banking tasks, use `send_message_to_remote_agent` to call a
        specific remote agent by its name.
        - Always base your answer on the actual tool results.
        - Always answer in Japanese.
        - When you delegate to a remote agent, clearly mention which agent you used in
        your final answer (e.g. 'Using Agentic Bank FAQ Agent: ...').
        - If the request is simple chit-chat or does not require any remote action,
        you may answer directly on your own.
        """,
        plugins=[a2a_plugin] if a2a_plugin is not None else [],
    )

    app.state.http_client = http_client
    app.state.a2a_client = a2a_client
    app.state.llama_agent = llama_agent
    app.state.qwen_agent = qwen_agent

    try:
        yield
    finally:
        await http_client.aclose()


app = FastAPI(
        title="Semantic Kernel Orchestrator",
        lifespan=lifespan
    )

def get_agent(request: ChatRequest) -> ChatCompletionAgent:
    return request.app.state.agent

@app.post("/orchestrate")
async def orchestrate(req: ChatRequest,) -> ChatResponse:
    if not req.messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    
    model = req.model or "llama"

    if model == "llama":
        agent = app.state.llama_agent
    elif model == "qwen":
        agent = app.state.qwen_agent
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")
    
    history = ChatHistory()
    for m in req.messages:
        if m.role == "user":
            history.add_user_message(m.content)
        elif m.role == "assistant":
            history.add_assistant_message(m.content)

    try:
        response_content = await agent.get_response(history)

        text = response_content.content
        if isinstance(text, list):
            text = "".join(map(str, text))
        if not isinstance(text, str):
            text = str(response_content)

        return ChatResponse(reply=text)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"agent_error: {type(e).__name__}: {e}",
        )
        
