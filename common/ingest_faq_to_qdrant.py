# scripts/ingest_bank_faq_qdrant.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from qdrant_client import QdrantClient, models

from embeddings import embed_texts


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "bank_faq"


CURRENT_DIR = Path(__file__).resolve().parent
DATA_PATH = CURRENT_DIR / "data" / "bank_faq.jsonl"


def main() -> None:
    client = QdrantClient(url=QDRANT_URL)

    path = Path("./data/bank_faq.jsonl")
    records = []
    for line in DATA_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))

    texts = [r["question"] + "\n" + r["answer"] for r in records]

    # Embeddingサーバにまとめて投げる
    vectors = embed_texts(texts)

    if not vectors:
        raise RuntimeError("No embeddings returned from embedding server")

    dim = len(vectors[0])

    # コレクションを作り直す
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=dim,
            distance=models.Distance.COSINE,
        ),
    )

    points = []
    for idx, rec in enumerate(records):
        text = rec["question"] + "\n" + rec["answer"]
        vec = vectors[idx]

        points.append(
            models.PointStruct(
                id=idx + 1,  # 🔸 ここを整数IDにする
                vector=vec,
                payload={
                    "faq_id": rec.get("id"),         # ← 元のIDは payload 側に退避
                    "question": rec["question"],
                    "answer": rec["answer"],
                    "category": rec.get("category", ""),
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Inserted {len(points)} FAQ entries into {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
