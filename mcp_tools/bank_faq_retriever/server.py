# mcp/bank_faq_retriever/server.py
from __future__ import annotations

import os
from typing import List, Dict, Any

from fastmcp import FastMCP
from qdrant_client import QdrantClient

from common.embeddings import embed_texts

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME = os.getenv("FAQ_COLLECTION", "bank_faq")

mcp = FastMCP("bank-faq-retriever")

# プロセス起動時にクライアント作成
_qdrant_client = QdrantClient(url=QDRANT_URL)


@mcp.tool()
async def search_bank_faq(
    query: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    エージェンティック銀行FAQからクエリに近いエントリを検索する。
    戻り値は question/answer/score を持つ dict のリスト。
    """
    # Embeddingサーバでクエリをベクトル化
    vectors = embed_texts([query])
    if not vectors:
        return []

    query_vec = vectors[0]

    # Qdrantでベクトル検索
    results = _qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=top_k,
        with_payload=True,
    )

    out: List[Dict[str, Any]] = []
    for r in results:
        payload = r.payload or {}
        out.append(
            {
                "score": r.score,
                "question": payload.get("question", ""),
                "answer": payload.get("answer", ""),
                "category": payload.get("category", ""),
            }
        )

    return out


if __name__ == "__main__":
    # FastMCPのHTTP/streamable HTTP トランスポートで起動
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8000,
        path="/mcp",
    )
