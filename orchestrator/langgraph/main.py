from contextlib import asynccontextmanager
import os
from typing import List, Literal, Optional, TypeAlias, Annotated, TypedDict

from fastapi import FastAPI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import BaseMessage, ToolMessage, ToolCall
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.graph import add_messages, StateGraph, START, END

from pydantic import BaseModel, TypeAdapter

# ======================================
# 環境変数
# ======================================
MODEL_ENDPOINT = os.getenv("MODEL_ENDPOINT")
MODEL_NAME = os.getenv("MODEL_NAME")
AGENTIC_BANK_FAQ_MCP_URL = os.getenv("AGENTIC_BANK_FAQ_MCP_URL", "http://bank_faq_retriever:8000/sse")


# ======================================
# グローバル（起動時に初期化）
# ======================================
mcp_client: MultiServerMCPClient | None = None
mcp_tools = []
tools_by_name: dict[str, any] = {}
model_with_tools: ChatOpenAI | None = None
graph = None


# ======================================
# HTTPリクエスト/レスポンス用の型
# ======================================
class Message(BaseModel):
    role: Literal["system", "user", "tool-return", "assistant", "tool-call"]
    content: str
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    tool_call_id: Optional[str] = None


MessageList: TypeAlias = list[Message]
Messages = TypeAdapter(MessageList)


# ======================================
# LangGraphのState 定義
# ======================================
class State(TypedDict):
    # add_messagesによって、各nodeが返したmessagesが
    # 既存のstate["messages"]にappendされていく
    messages: Annotated[List[BaseMessage], add_messages]


# ======================================
# LangGraphのnode定義
# ======================================
async def call_model(state: State) -> dict:
    """
    LLMを呼び出すnode。
    state["messages"]にはSystem+過去メッセージ+ToolMessageなど。
    """
    if model_with_tools is None:
        raise RuntimeError("model_with_tools is not initialized")

    llm_response = await model_with_tools.ainvoke(state["messages"])
    
    return {"messages": [llm_response]}


async def call_tools(state: State) -> dict:
    """
    直近のAIMessageに含まれるtool_callsを見て、
    MCPツールを実行し、その結果をToolMessageとして返すnode。
    """
    if not state["messages"]:
        return {"messages": []}

    last_message = state["messages"][-1]

    tool_calls: List[ToolCall] = getattr(last_message, "tool_calls", []) or []
    if not tool_calls:
        return {"messages": []}

    results: List[ToolMessage] = []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool = tools_by_name.get(tool_name)
        if tool is None:
            continue

        observation = await tool.ainvoke(tool_call)

        results.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call["id"],
            )
        )

    return {"messages": results}


def route_model_output(state: State) -> str:
    """
    call_modelの出力に基づいて分岐させるルーター関数。
    - tool_callsがあれば"tools" へ
    - なければ"end" へ
    """
    if not state["messages"]:
        return "end"

    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", []) or []

    if tool_calls:
        return "tools"
    return "end"


# ======================================
# FastAPI lifespan: 起動時にMCP & LangGraphを初期化
# ======================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global mcp_client, mcp_tools, tools_by_name, model_with_tools, graph

    # MCPクライアント初期化
    mcp_client = MultiServerMCPClient(
        {
            "bank_faq_tools": {
                "transport": "sse",
                "url": AGENTIC_BANK_FAQ_MCP_URL,
            }
        }
    )

    mcp_tools = await mcp_client.get_tools()
    tools_by_name = {t.name: t for t in mcp_tools}

    # LLM初期化&ツールバインド
    base_model = ChatOpenAI(
        base_url=MODEL_ENDPOINT,
        api_key="this is dummy.",
        model=MODEL_NAME,
    )
    model_with_tools = base_model.bind_tools(mcp_tools)

    # State初期化
    graph_builder = StateGraph(State)

    # node登録
    graph_builder.add_node("model", call_model)
    graph_builder.add_node("tools", call_tools)

    # edge設定
    graph_builder.add_edge(START, "model")  # 開始 → LLM

    # LLMの出力に応じて分岐
    graph_builder.add_conditional_edges(
        "model",
        route_model_output,
        {
            "tools": "tools",  # route_model_outputが"tools"を返したらcall_tools へ
            "end": END,        # "end"の場合は終了
        },
    )

    graph_builder.add_edge("tools", "model")

    
    graph = graph_builder.compile()
    
    try:
        yield
    finally:
    
        if mcp_client is not None:
            await mcp_client.aclose()


app = FastAPI(lifespan=lifespan)


# ======================================
# /v1/chat/completions エンドポイント
# ======================================
@app.post("/v1/chat/completions")
async def orchestrate(messages_request: MessageList) -> List[Message]:
    
    print(f"messages_request: {messages_request}")
    print(f"user_message: {messages_request[0].content if messages_request else ''}")
    
    lc_messages: List[BaseMessage] = []

    system_msg = SystemMessage(
        content="""
        You are an AI assistant that has access to the Agentic-Bank FAQ.
        When you are asked about bank-specific information such as products, services, fees, or procedures,
        you must always use the tool (bank_faq_tools) to retrieve the information before answering.
        Do not guess or answer based solely on your own knowledge.
        """
    )
    lc_messages.append(system_msg)

    for m in messages_request:
        if m.role == "user":
            lc_messages.append(HumanMessage(content=m.content))


    if graph is None:
        raise RuntimeError("Graph is not initialized")

    final_state: State = await graph.ainvoke({"messages": lc_messages})
    history = final_state["messages"]
    print(f"history: {history}")

    last_ai_message: Optional[AIMessage] = None
    for m in reversed(history):
        print(f"m: {m}")
        if isinstance(m, AIMessage):
            last_ai_message = m
            break

    if last_ai_message is None:
        return []

    response_messages = [
        Message(role="assistant", content=last_ai_message.content)
    ]
    return response_messages