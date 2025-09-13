"""
Main Orchestrator Agent (non-destructive)
-------------------------------------------------------------------------------
Provides a clean entrypoint that wires the three agents together without
modifying your original scripts. This file is NEW and safe to add.

Pipeline:
    RetrievalAgent  -> RerankerAgent -> AnswererAgent

Features:
- Mirrors your current end-to-end behavior (same models/prompts/index params)
- Optional feedback logging (CSV) like your existing main.py
- Simple CLI usage for quick tests

Usage:
    python main_agent.py --query "Can I return engraved items?" --top_k 3

Note:
    Requires Milvus running and collection `chunk_embeddings` loaded
    (same as your current setup).
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

from retrieval_agent import RetrievalAgent
from answerer_agent import AnswererAgent
# Use the finetuned reranker (LoRA + regression head)
from reranker_agent_finetuned import RerankerAgentFineTuned as RerankerAgent, FTConfig as RerankerConfig


class MainAgent:
    def __init__(
        self,
        lora_dir: str = "bge_reranker_Final",     # directory with adapters + regression_head.pt
        score_mode: str = "sigmoid_0_5",          # {"sigmoid_0_5","sigmoid_0_1"}
        feedback_csv: str = "feedback_log.csv",
    ) -> None:
        # Initialize sub-agents
        self.retriever = RetrievalAgent()
        self.reranker = RerankerAgent(
            RerankerConfig(
                lora_dir=lora_dir,
                score_mode=score_mode,
            )
        )
        # self.answerer = AnswererAgent()
        self.answerer = AnswererAgent(model_name="microsoft/Phi-3-mini-4k-instruct")
        self.feedback_csv = feedback_csv

    def run(self, query: str, top_k: int = 3, ask_feedback: bool = False) -> str:
        # 1) Retrieve
        df_candidates = self.retriever.retrieve(query, initial_k=top_k * 10)

        # 2) Rerank
        df_reranked = self.reranker.rerank(query, df_candidates)

        # 3) Answer
        final_answer = self.answerer.answer(query, df_reranked, top_k=top_k)

        # 4) Optional feedback (interactive)
        if ask_feedback:
            print("\n Rate this answer from 1 to 5:")
            print("1 = Poor • 2 = Fair • 3 = Good • 4 = Very Good • 5 = Excellent")
            while True:
                fb = input("Your rating: ").strip()
                if fb in {"1", "2", "3", "4", "5"}:
                    break
                print(" Please enter a number from 1 to 5.")
            self._log_feedback(query, df_reranked, final_answer, fb)
            print(" Feedback saved. Thank you!")

        return final_answer

    def _log_feedback(self, query, df_reranked, answer, feedback: str) -> None:
        timestamp = datetime.now().isoformat()
        top_chunks = " || ".join(df_reranked["chunk_text"].head(3).astype(str).tolist())
        path = Path(self.feedback_csv)
        header_needed = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if header_needed:
                writer.writerow(["timestamp", "query", "top_chunks", "answer", "feedback"])
            writer.writerow([timestamp, query, top_chunks, answer, feedback])


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ReturnMind Main Agent Orchestrator")
    p.add_argument("--query", type=str, required=True, help="User question")
    p.add_argument("--top_k", type=int, default=3, help="How many chunks to pass to answerer")
    p.add_argument("--no_feedback", action="store_true", help="Disable interactive feedback prompt")
    # Use --lora_dir to match the finetuned agent
    p.add_argument("--lora_dir", type=str, default="bge_reranker_Final",
                   help="LoRA directory with adapter_model.safetensors, adapter_config.json, regression_head.pt")
    p.add_argument("--score_mode", type=str, default="sigmoid_0_5",
                   choices=["sigmoid_0_5", "sigmoid_0_1"], help="Reranker score scaling")
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    app = MainAgent(lora_dir=args.lora_dir, score_mode=args.score_mode)
    answer = app.run(args.query, top_k=args.top_k, ask_feedback=not args.no_feedback)
    print("\n--------------------- Final Answer ---------------------\n")
    print(answer)
