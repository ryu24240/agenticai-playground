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
            "Send a user message to a remote A2A agent by name and return the "
            "agent's textual responses."
        ),
    )
    async def send_message_to_remote_agent(
        self,
        agent_name: Annotated[str, "Name property of the remote AgentCard"],
        message: Annotated[str, "User message to send to the remote agent."],
    ) -> Annotated[str, "Concatenated text reply from the remote agent"]:
        """
        名前で指定されたリモートエージェントにメッセージを送り、
        text パートを連結して返す。
        """
        client = self._a2a_client.clients.get(agent_name)
        if client is None:
            raise ValueError(f"Agent {agent_name!r} not found in A2A clients")

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
