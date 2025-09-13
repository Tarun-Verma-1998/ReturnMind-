"""
Retrieval Agent for ReturnMind (non-destructive, drop-in module)
-----------------------------------------------------------------------------
Keeps the original retrieval logic intact while packaging it in an agent class.
- Embedding model: BAAI/bge-base-en-v1.5 (same as current project)
- Query embedding: CLS token, L2-normalized (as in the current code)
- Milvus search: COSINE metric, HNSW index, ef=64 (same defaults)
- Output: pandas.DataFrame with [chunk_text, doc_name, chunk_id, cosine_score]

This file is NEW and does not modify existing files.
"""
from __future__ import annotations

import os
from typing import List, Optional

import torch
import pandas as pd
from pymilvus import connections, Collection
from transformers import AutoTokenizer, AutoModel


class RetrievalAgent:
    """Agent that retrieves candidate chunks from Milvus for a given query.

    Design goals
    ------------
    1) **Non-destructive:** mirrors your existing retrieval logic so behavior doesn't change.
    2) **Self-contained:** manages its own model loading and Milvus connection.
    3) **Explicit params:** `initial_k` and `ef` are configurable, with safe defaults.
    """

    def __init__(
        self,
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
        collection_name: str = "chunk_embeddings",
        embed_model_name: str = "BAAI/bge-base-en-v1.5",
        device: Optional[str] = None,
    ) -> None:
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.collection_name = collection_name
        self.embed_model_name = embed_model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Lazy-loaded on first use
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModel] = None
        self._collection: Optional[Collection] = None

    # -----------------------------
    # Private helpers
    # -----------------------------
    def _ensure_milvus(self) -> None:
        # Connect only once per process; reuse the default alias
        if not connections.has_connection(alias="default"):
            connections.connect(alias="default", host=self.milvus_host, port=self.milvus_port)
        if self._collection is None:
            self._collection = Collection(self.collection_name)
            # Safe to call multiple times; Milvus will handle caching
            self._collection.load()

    def _ensure_model(self) -> None:
        if self._tokenizer is None or self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.embed_model_name)
            self._model = AutoModel.from_pretrained(self.embed_model_name).to(self.device)
            self._model.eval()

    def _embed_query(self, text: str) -> List[float]:
        """Embed the query exactly like the current pipeline (CLS, L2 norm)."""
        assert self._tokenizer is not None and self._model is not None, "Call _ensure_model() first"
        with torch.no_grad():
            tokens = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            output = self._model(**tokens)
            # CLS-style token (first token of last hidden state), same as current code
            query_embedding = output.last_hidden_state[:, 0, :]
            # L2 normalize for cosine similarity
            query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
            return query_embedding.cpu().numpy().tolist()[0]

    # -----------------------------
    # Public API
    # -----------------------------
    def retrieve(
        self,
        query: str,
        initial_k: int = 30,
        ef: int = 64,
        output_fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Retrieve top `initial_k` candidates from Milvus.

        Parameters
        ----------
        query : str
            The user question to embed and search.
        initial_k : int, default=30
            How many candidates to fetch for downstream reranking (top_k * 10 in original).
        ef : int, default=64
            HNSW ef search parameter (same as current code).
        output_fields : list of str, optional
            Extra fields to request from Milvus. Defaults to ["chunk_text", "doc_name", "chunk_id"].

        Returns
        -------
        pd.DataFrame with columns: [chunk_text, doc_name, chunk_id, cosine_score]
        """
        self._ensure_milvus()
        self._ensure_model()

        # Prepare query vector
        vector = self._embed_query(query)

        # Default fields mirror the existing code
        if output_fields is None:
            output_fields = ["chunk_text", "doc_name", "chunk_id"]

        # Milvus search params (same metric + ef)
        search_params = {"metric_type": "COSINE", "params": {"ef": ef}}

        results = self._collection.search(
            data=[vector],
            anns_field="embedding",
            param=search_params,
            limit=initial_k,
            output_fields=output_fields,
        )

        # Convert to DataFrame with the exact same columns as before
        retrieved = []
        for hit in results[0]:
            retrieved.append({
                "chunk_text": hit.entity.get("chunk_text"),
                "doc_name": hit.entity.get("doc_name"),
                "chunk_id": hit.entity.get("chunk_id"),
                # Milvus returns distance; with COSINE, higher is more similar in this config
                "cosine_score": hit.distance,
            })

        df = pd.DataFrame(retrieved)
        return df


# Optional local sanity check
if __name__ == "__main__":
    agent = RetrievalAgent()
    try:
        df_candidates = agent.retrieve("can you explain original payment method for refund?", initial_k=30)
        print(df_candidates.head(5))
    except Exception as e:
        print("[RetrievalAgent] Sanity check failed:", e)
