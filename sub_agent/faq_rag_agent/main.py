# ai_orchestrator/subagent/mock_agent_server/main.py
from __future__ import annotations

import os

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message

from openai import AsyncOpenAI
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion, OllamaChatPromptExecutionSettings
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.mcp import MCPSsePlugin
from semantic_kernel.contents import AuthorRole, ChatMessageContent, ChatHistory
from semantic_kernel.agents import ChatCompletionAgent

# from fastmcp import Client as MCPClient

AGENTIC_BANK_FAQ_MCP_URL = os.getenv("AGENTIC_BANK_FAQ_MCP_URL", "http://bank_faq_retriever:8000/mcp")




LLAMA_ENDPOINT = os.getenv("LLM_URL", "http://localhost:11434")
QWEN_ENDPOINT = os.getenv("MODEL_ENDPOINT")
QWEN_MODEL_NAME = os.getenv("MODEL_NAME")
OPENAI_API_KEY = "this is dummy"

SYSTEM_PROMPT = """
あなたは"エージェンティック銀行"向けのサブエージェントです。
このエージェントは、エージェンティック銀行の公開FAQに対して日本語で分かりやすく説明する役割を持ちます。

- 回答は必ず日本語で行ってください。
- 分からないことは無理に断定せず、「推測」「一般的には〜」といった形で慎重に回答してください。
"""


class SKSubAgentExecutor(AgentExecutor):
    """
    A2A AgentExecutor実装
    """

    def __init__(self) -> None:
        async_openai = AsyncOpenAI(
            base_url=LLAMA_ENDPOINT,
            api_key=OPENAI_API_KEY,
        )

        llama_service = OllamaChatCompletion(
            ai_model_id="llama3.1:latest", host=LLAMA_ENDPOINT
        )
        qwen_service = OpenAIChatCompletion(
            ai_model_id=QWEN_MODEL_NAME,
            api_key=OPENAI_API_KEY,
            async_client=async_openai,
        )

        self.faq_mcp_plugin = MCPSsePlugin(
            name="Agentic_Bank_FAQ",
            description="Retrieval Agentic-Bank FAQ from Vector DB.",
            url=AGENTIC_BANK_FAQ_MCP_URL,
            load_prompts=False,
        )

        kernel = Kernel()
        kernel.add_service(llama_service)
        kernel.add_service(qwen_service)
        kernel.add_plugin(self.faq_mcp_plugin)

        self.llama_agent = ChatCompletionAgent(
            service=llama_service,
            name="llama-faq-subagent",
            kernel=kernel,
            instructions=SYSTEM_PROMPT,
            plugins=[self.faq_mcp_plugin],
        )

        self.qwen_agent = ChatCompletionAgent(
            service=qwen_service,
            name="qwen-faq-subagent",
            kernel=kernel,
            instructions=SYSTEM_PROMPT,
            plugins=[self.faq_mcp_plugin],
        )

        # LLMサービスの設定を反映
        self.llama_agent.configure_service()
        self.qwen_agent.configure_service()

        # 接続状態を追跡するフラグ
        self._plugin_connected = False

    async def _ensure_plugin_connected(self):
        """MCPプラグインが未接続なら接続する。"""
        if not self._plugin_connected:
            await self.faq_mcp_plugin.connect()
            self._plugin_connected = True

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """
        A2A サーバからのリクエストを受け取り、
        SK エージェントを呼び出して結果を A2A イベントとして返す。
        """
        user_input = context.get_user_input() or ""

        print("[faq_rag_agent] context =", context)

        history = ChatHistory()
        history.add_user_message(user_input)

        # メタデータからモデル種別を取得
        model = None
        if hasattr(context, "metadata") and context.metadata:
            model = context.metadata.get("model")

        print(f"[faq_rag_agent] model={model}, user_input={user_input}")

        # MCPプラグインに接続
        await self._ensure_plugin_connected()

        agent = self.qwen_agent if model == "qwen" else self.llama_agent

        response_content = await agent.get_response(history)

        text = response_content.content
        if isinstance(text, list):
            text = "".join(map(str, text))
        if not isinstance(text, str):
            text = str(text)

        await event_queue.enqueue_event(new_agent_text_message(text))

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        raise Exception("cancel not supported")

skill = AgentSkill(
    id="mock_bank_subagent",
    name="Mock Bank SubAgent",
    description="エージェンティック銀行向けチャットボットのサブエージェント。公開FAQの説明を行う。",
    tags=["agentic-bank", "FAQ", "semantic-kernel"],
    examples=[
        "振込の手順を教えて",
        "subagent: テストとしてこのサブエージェントに話しかけています",
    ],
)

public_agent_card = AgentCard(
    name="Agentic-Bank FAQ SubAgent",
    description="Semantic Kernel ベースで実装されたエージェンティック銀行向けFAQサブエージェント。",
    url=os.getenv("PUBLIC_AGENT_URL", "http://mock_agent_server:8200/"),
    version="1.0.0",
    default_input_modes=["text"],
    default_output_modes=["text"],
    capabilities=AgentCapabilities(streaming=False),
    skills=[skill],
)

request_handler = DefaultRequestHandler(
    agent_executor=SKSubAgentExecutor(),
    task_store=InMemoryTaskStore(),
)

server = A2AStarletteApplication(
    agent_card=public_agent_card,
    http_handler=request_handler,
)

app = server.build()
