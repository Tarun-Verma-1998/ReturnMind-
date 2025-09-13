"""
Reranker Agent for ReturnMind (non-destructive)
-------------------------------------------------------------------------------
Wraps BGE Cross-Encoder (optionally with LoRA) to score (query, chunk) pairs.
Keeps logic equivalent to your existing `rerank_with_finetuned_model.py`:
- Cross-encoder scoring over (query, passage) pairs
- Optional LoRA adapter loading for your fine-tuned reranker
- Optional regression mapping to [0, 5] via sigmoid * 5 (if your head was trained that way)
- Returns a DataFrame with the same rows + `rerank_score`, sorted desc

This file is NEW and does not modify existing files.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, List

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False


@dataclass
class RerankerConfig:
    base_model_name: str = "BAAI/bge-reranker-base"
    lora_adapter_path: Optional[str] = None  # directory with LoRA adapter if used
    device: Optional[str] = None  # auto-detect if None
    max_length: int = 512
    batch_size: int = 16
    # If your fine-tune used a regression head mapped to 0..5 via sigmoid*5
    score_mode: str = "raw"  # one of {"raw", "sigmoid_0_5"}


class RerankerAgent:
    """Cross-encoder reranker for (query, chunk) pairs.

    Usage:
        agent = RerankerAgent(RerankerConfig(lora_adapter_path="./lora_reranker", score_mode="sigmoid_0_5"))
        df_reranked = agent.rerank(query, df_candidates)
    """

    def __init__(self, config: Optional[RerankerConfig] = None) -> None:
        self.cfg = config or RerankerConfig()
        self.device = self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForSequenceClassification] = None

    # -----------------------------
    # Private helpers
    # -----------------------------
    def _ensure_model(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.base_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.cfg.base_model_name)

        if self.cfg.lora_adapter_path:
            if not _HAS_PEFT:
                raise RuntimeError("peft is not installed but lora_adapter_path was provided")
            if not os.path.isdir(self.cfg.lora_adapter_path):
                raise FileNotFoundError(f"LoRA adapter path not found: {self.cfg.lora_adapter_path}")
            model = PeftModel.from_pretrained(model, self.cfg.lora_adapter_path)

        self._model = model.to(self.device)
        self._model.eval()

    def _batched(self, arr: List[str], batch: int) -> List[List[str]]:
        return [arr[i : i + batch] for i in range(0, len(arr), batch)]

    def _map_score(self, logits: torch.Tensor) -> torch.Tensor:
        """Map logits to final scores depending on configured mode."""
        if self.cfg.score_mode == "sigmoid_0_5":
            return torch.sigmoid(logits) * 5.0
        # default: raw logits as scores (monotonic ranking)
        return logits

    # -----------------------------
    # Public API
    # -----------------------------
    def rerank(self, query: str, df_candidates: pd.DataFrame) -> pd.DataFrame:
        """Return a new DataFrame with a `rerank_score` column, sorted desc.

        Expects df_candidates to include at least a `chunk_text` column.
        Keeps all original columns and row order is re-sorted by score.
        """
        if df_candidates is None or df_candidates.empty:
            return df_candidates
        if "chunk_text" not in df_candidates.columns:
            raise ValueError("df_candidates must contain a 'chunk_text' column")

        self._ensure_model()
        assert self._tokenizer is not None and self._model is not None

        passages = df_candidates["chunk_text"].astype(str).tolist()
        scores: List[float] = []

        with torch.no_grad():
            for batch_passages in self._batched(passages, self.cfg.batch_size):
                enc = self._tokenizer(
                    [query] * len(batch_passages),
                    batch_passages,
                    truncation=True,
                    max_length=self.cfg.max_length,
                    padding=True,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out = self._model(**enc)
                logits = out.logits.view(-1)
                final_scores = self._map_score(logits)
                scores.extend(final_scores.detach().cpu().tolist())

        df_out = df_candidates.copy()
        df_out["rerank_score"] = scores
        df_out = df_out.sort_values("rerank_score", ascending=False).reset_index(drop=True)
        return df_out


# Optional local sanity check
if __name__ == "__main__":
    import numpy as np
    sample = pd.DataFrame({
        "chunk_text": [
            "Refunds will be issued to the original payment method within 5–7 days.",
            "Holiday returns must be made within 30 days of purchase.",
        ],
        "doc_name": ["policy.md", "policy.md"],
        "chunk_id": [0, 1],
        "cosine_score": [0.77, 0.65],
    })

    cfg = RerankerConfig(
        base_model_name="BAAI/bge-reranker-base",
        lora_adapter_path=None,  # set your LoRA folder if applicable
        score_mode="raw",       # or "sigmoid_0_5" to mimic 0..5 regression
    )
    agent = RerankerAgent(cfg)
    df = agent.rerank("can you explain original payment method for refund?", sample)
    print(df)
