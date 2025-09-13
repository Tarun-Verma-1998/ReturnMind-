"""
Reranker Agent (Fine-Tuned Head)
-------------------------------------------------------------------------------
Loads your fine-tuned BGE cross-encoder with LoRA adapters **and** the trained
regression head saved as `regression_head.pt`.

- Base encoder: BAAI/bge-reranker-base (cross-encoder)
- LoRA adapters: load from directory (e.g., bge_reranker_Final/)
- Regression head: same shape as during training; weights loaded from
  bge_reranker_Final/regression_head.pt
- Output scores in [0,1] by default (sigmoid); option to map to [0,5].

This file is NEW and does not modify existing files.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, List

import torch
import pandas as pd
from torch import nn
from transformers import AutoTokenizer, AutoModel

try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False


@dataclass
class FTConfig:
    lora_dir: str = "bge_reranker_Final"  # directory with adapter_model.safetensors + adapter_config.json
    base_model_name: str = "BAAI/bge-reranker-base"
    device: Optional[str] = None
    max_length: int = 512
    batch_size: int = 16
    score_mode: str = "sigmoid_0_5"  # {"sigmoid_0_5", "sigmoid_0_1"}


class RerankerRegressor(nn.Module):
    """Encoder + small regression head with sigmoid (same as training)."""
    def __init__(self, base: AutoModel, hidden_size: int):
        super().__init__()
        self.base = base
        self.regression = nn.Linear(hidden_size, 1)
        self.activation = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        score = self.regression(cls)
        return self.activation(score).squeeze(-1)  # [B]


class RerankerAgentFineTuned:
    """Uses LoRA adapters + trained regression head weights for reranking."""

    def __init__(self, cfg: Optional[FTConfig] = None) -> None:
        self.cfg = cfg or FTConfig()
        self.device = self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[RerankerRegressor] = None

    # -----------------------------
    # Private helpers
    # -----------------------------
    def _ensure_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        if not os.path.isdir(self.cfg.lora_dir):
            raise FileNotFoundError(f"LoRA directory not found: {self.cfg.lora_dir}")
        if not _HAS_PEFT:
            raise RuntimeError("peft is not installed. pip install peft")

        # Tokenizer comes from the LoRA dir (you saved it there)
        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.lora_dir)

        # Load base encoder, then apply LoRA adapters from folder
        base = AutoModel.from_pretrained(self.cfg.base_model_name)
        base = PeftModel.from_pretrained(base, self.cfg.lora_dir)
        base.eval()

        # Build head and load weights
        hidden = base.base_model.config.hidden_size if hasattr(base, "base_model") else base.config.hidden_size
        model = RerankerRegressor(base, hidden)

        head_path = os.path.join(self.cfg.lora_dir, "regression_head.pt")
        if not os.path.isfile(head_path):
            raise FileNotFoundError(f"Missing regression head weights at: {head_path}")
        sd = torch.load(head_path, map_location="cpu")
        model.regression.load_state_dict(sd)
        model.to(self.device)
        model.eval()
        self._model = model

    def _batched(self, arr: List[str], batch: int) -> List[List[str]]:
        return [arr[i : i + batch] for i in range(0, len(arr), batch)]

    def _map_score(self, scores: torch.Tensor) -> torch.Tensor:
        if self.cfg.score_mode == "sigmoid_0_5":
            return scores * 5.0
        return scores  # sigmoid already in [0,1]

    # -----------------------------
    # Public API
    # -----------------------------
    def rerank(self, query: str, df_candidates: pd.DataFrame) -> pd.DataFrame:
        if df_candidates is None or df_candidates.empty:
            return df_candidates
        if "chunk_text" not in df_candidates.columns:
            raise ValueError("df_candidates must contain a 'chunk_text' column")

        self._ensure_model()
        assert self._model is not None and self._tokenizer is not None

        passages = df_candidates["chunk_text"].astype(str).tolist()
        scores_all: List[float] = []

        with torch.no_grad():
            for batch_passages in self._batched(passages, self.cfg.batch_size):
                enc = self._tokenizer(
                    [f"Query: {query} Document: {p}" for p in batch_passages],
                    truncation=True,
                    max_length=self.cfg.max_length,
                    padding=True,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                scores = self._model(**enc)  # sigmoid in [0,1]
                scores = self._map_score(scores)
                scores_all.extend(scores.detach().cpu().tolist())

        df_out = df_candidates.copy()
        df_out["rerank_score"] = scores_all
        df_out = df_out.sort_values("rerank_score", ascending=False).reset_index(drop=True)
        return df_out


# Optional sanity check
if __name__ == "__main__":
    import pandas as pd

    sample = pd.DataFrame({
        "chunk_text": [
            "Refunds will be issued to the original payment method within 5–7 days.",
            "Holiday returns must be made within 30 days of purchase.",
        ],
        "doc_name": ["policy.md", "policy.md"],
        "chunk_id": [0, 1],
        "cosine_score": [0.77, 0.65],
    })

    agent = RerankerAgentFineTuned(FTConfig(lora_dir="bge_reranker_Final", score_mode="sigmoid_0_5"))
    df = agent.rerank("can you explain original payment method for refund?", sample)
    print(df)
