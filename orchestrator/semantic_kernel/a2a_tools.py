from __future__ import annotations
from typing import List, Annotated

import json
import uuid

from semantic_kernel.functions import kernel_function

from a2a.types import Message as A2AMessage, Part, Role
from a2a.types import TextPart as A2ATextPart
from a2aclient.client import A2AClient



class A2ATools:
    """
    ChatCompletionAgent から呼び出される「ツール」としての A2A プラグイン。

    - list_remote_agents()
        -> 利用可能なリモートエージェント一覧を返す
    - send_message_to_remote_agent(agent_name, message)
        -> 指定したリモートエージェントに A2A でメッセージ送信
    """

    def __init__(self, a2a_client: A2AClient):
        self._a2a_client = a2a_client

    @kernel_function(
        name="list_remote_agents",
        description="List available remote A2A agents and their descriptions.",
    )
    def list_remote_agents(self) -> Annotated[str, "JSON array of {name, description}"]:
        """
        利用可能なリモートエージェントの一覧をJSON文字列で返す。
        """
        info = [
            {"name": card.name, "description": card.description}
            for card in self._a2a_client.cards
        ]
        return json.dumps(info, ensure_ascii=False)
    

    @kernel_function(
        name="send_message_to_remote_agent",
        description=(
            "指定された remote A2A agent に『ユーザーの質問テキストそのもの』を渡して実行します。"
            "parameters.message には必ず「ユーザーが実際に入力した質問」を日本語でそのまま入れてください。"
            "要約や別表現、暗号化した文章ではなく、できるだけ原文に近いテキストを渡してください。"
        ),
    )
    async def send_message_to_remote_agent(
        self,
        agent_name: Annotated[str, "Name property of the remote AgentCard"],
        message: Annotated[str, "User message to send to the remote agent."],
    ) -> Annotated[str, "Concatenated text reply from the remote agent"]:

        print(f"[A2ATools] send_message_to_remote_agent called. "
            f"agent_name={agent_name!r}, message={message!r}")
        
        if self._a2a_client is None:
            raise ValueError("A2A client is not initialized")

        client = self._a2a_client.clients.get(agent_name)

        if client is None:
            available = list(self._a2a_client.clients.keys())
            print(f"[A2ATools] agent not found: {agent_name!r}, available={available}")

            if len(available) == 1:
                print("[A2ATools] fallback: using the only available agent.")
                client = self._a2a_client.clients[available[0]]

            if client is None:
                raise ValueError(
                    f"Agent {agent_name!r} not found in A2A clients. "
                    f"available={available}"
                )

        request_message = A2AMessage(
            role=Role.user,
            parts=[Part(root=A2ATextPart(text=message))],
            message_id=str(uuid.uuid4()),
        )

        texts: list[str] = []
        async for response in client.send_message(request_message):
            for part in response.parts:
                if part.root.kind == "text":
                    texts.append(part.root.text)

        if not texts:
            return "Remote agent returned no text parts."

        return "\n".join(texts)
