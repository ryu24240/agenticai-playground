# ai_orchestrator/semantic_kernel/a2aclient/__init__.py
from __future__ import annotations

import asyncio
from typing import Iterable

import httpx

from a2a.client import A2ACardResolver, Client, ClientConfig, ClientFactory
from a2a.types import AgentCard, TransportProtocol


class A2AClient:
    """
    A2A SDK を利用してリモートエージェントを発見し、
    AgentCard とそれに紐づく Client を保持する薄いラッパークラス。

    - create(): リモートエージェントの URL 一覧から AgentCard / Client をまとめて生成
    - cards: 取得した AgentCard のリスト
    - clients: {agent_name: Client} の辞書
    """

    @classmethod
    async def create(
        cls,
        remote_agent_addresses: Iterable[str],
        httpx_client: httpx.AsyncClient,
    ) -> "A2AClient":
        """
        非同期に A2AClient を構築するファクトリメソッド。

        1. remote_agent_addresses から AgentCard を取得
        2. AgentCard から A2A Client を作成
        """
        self = cls()

        self.cards: list[AgentCard] = await self._gather_agent_cards(
            remote_agent_addresses
        )

        self.clients: dict[str, Client] = await self._gather_clients(
            self.cards, httpx_client
        )

        return self

    # ----------------- 内部処理 -----------------

    async def _gather_agent_cards(self, addresses: Iterable[str]) -> list[AgentCard]:
        """
        複数の remote URL から AgentCard を並列取得。
        """
        async with asyncio.TaskGroup() as tg:
            tasks: list[asyncio.Task[AgentCard]] = [
                tg.create_task(self._retrieve_agent_card(addr)) for addr in addresses
            ]
        cards: list[AgentCard] = [task.result() for task in tasks]
        return cards

    async def _retrieve_agent_card(self, address: str) -> AgentCard:
        """
        単一の remote URL から AgentCard を取得。
        """
        async with httpx.AsyncClient() as tmp_client:
            card_resolver = A2ACardResolver(tmp_client, address)
            card: AgentCard = await card_resolver.get_agent_card()
        return card

    async def _gather_clients(
        self,
        cards: Iterable[AgentCard],
        httpx_client: httpx.AsyncClient,
    ) -> dict[str, Client]:
        """
        AgentCard のコレクションから A2A Client を並列生成。
        """
        clients: dict[str, Client] = {}

        async with asyncio.TaskGroup() as tg:
            tasks: dict[str, asyncio.Task[Client]] = {
                card.name: tg.create_task(self._retrieve_client(card, httpx_client))
                for card in cards
            }

        for name, task in tasks.items():
            clients[name] = task.result()

        return clients

    async def _retrieve_client(
        self, card: AgentCard, httpx_client: httpx.AsyncClient
    ) -> Client:
        """
        AgentCard から A2A Client を生成。
        """
        config = ClientConfig(
            httpx_client=httpx_client,
            supported_transports=[
                TransportProtocol.jsonrpc,
                TransportProtocol.http_json,
            ],
        )
        client_factory = ClientFactory(config)
        client: Client = client_factory.create(card)
        return client
