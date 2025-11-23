from langchain.messages import ( SystemMessage, HumanMessage, ToolCall)
from langchain.tools import tool
from langchain.chat_models import init_chat_model 
from langchain_core.messages import BaseMessage

from langgraph.graph import add_messages
from langgrahph.func import entrypoint, task

from pydantic import BaseModel, Field
from typing import List

model = init_chat_model(
    "llama3.1:latest",
    model_provider="ollama",
    temperature=0.7,
)

@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

tools = [add]
tools_by_name = {tool.name: tool for tool in tools}

model_with_tools = model.bind_tools(tools)

@task
def call_llm(messages: list[BaseMessage]):
    """LLM decides whether to call a tool or not"""
    return model_with_tools.invoke(
        [
            SystemMessage(
                content="You are an AI assistant that helps people find information."
            )
        ]
        + messages
    )
    
@task
def call_tool(tool_call: ToolCall):
    """Call the tool specified by the LLM"""
    tool = tools_by_name[tool_call["name"]]
    return tool.invoke(tool_call)

@entrypoint()
def agent(messages: List[BaseMessage]):
    model_response = call_llm(messages).result()
    
    while True:
        if not model_response.tool_calls:
            break
        
        tool_result_futures = [
            call_tool(tool_call) for tool_call in model_response.tool_calls
        ]
        tool_results = [fut.result() for fut in tool_result_futures]
        messages = add_messages(messages, [model_response, *tool_results])
        model_response = call_llm(messages).result()
        
        
    