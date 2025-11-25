# common/embeddings.py
from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv

import requests

load_dotenv('.env')

EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", "http://xxxx")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "xxxx")


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    オンプレの Embedding サーバに HTTP で問い合わせてベクトルを取得する。
    ここでは OpenAI 互換 /v1/embeddings を想定。
    カスタム API の場合はこの関数内だけ書き換える。
    """
    if not texts:
        return []

    url = f"{EMBEDDING_API_BASE.rstrip('/')}/v1/embeddings"
    payload = {
        "model": EMBEDDING_MODEL_ID,
        "input": texts,
    }

    headers = {
        "Content-Type": "application/json",
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    embeddings: List[List[float]] = []
    for item in data["data"]:
        embeddings.append(item["embedding"])

    return embeddings
