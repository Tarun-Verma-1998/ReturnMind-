"""
Answerer Agent for ReturnMind (non-destructive)
-------------------------------------------------------------------------------
Wraps your existing Mistral-7B answer generation logic from `answer_generation.py`.
- Uses the SAME model, prompt, and formatting.
- Accepts the reranked DataFrame and a top_k.
- Lazily loads tokenizer/model once and reuses them.

This file is NEW and does not modify existing files.
"""
from __future__ import annotations

import torch
import pandas as pd
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM


class AnswererAgent:
    """Agent that produces a grounded answer using reranked chunks and Mistral-7B.

    Usage:
        agent = AnswererAgent()
        answer = agent.answer(query, df_reranked, top_k=3)
    """

    def __init__(
        self,
        # model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        model_name: str = "microsoft/phi-3-mini-4k-instruct",
        # model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForCausalLM] = None

    # -----------------------------
    # Private helpers
    # -----------------------------
    def _ensure_model(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return

        is_cuda = torch.cuda.is_available()
        self.device = "cuda" if is_cuda else "cpu"

        print(f"AnswererAgent: loading model -> {self.model_name}")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        if is_cuda:
            # GPU path
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            # CPU path — no device_map; stream weights to reduce peak RAM
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to("cpu")

        self._model.eval()


    def _build_prompt(self, query: str, top_chunks: list[str]) -> str:
        # EXACT prompt style from answer_generation.py
        context = "\n".join([f"- {chunk}" for chunk in top_chunks])
        prompt = f"""[INST] You are a helpful assistant answering customer queries using the company's return policy.
Context:
{context}

Question: {query}
Answer using 1–2 short sentences. Be concise but cover all important conditions from the most relevant passage. [/INST]
"""
        return prompt

    # -----------------------------
    # Public API
    # -----------------------------
    def answer(self, query: str, reranked_df: pd.DataFrame, top_k: int = 3, print_chunks: bool = True) -> str:
        """Generate the grounded answer using the top-k reranked chunks.

        Mirrors the behavior of `generate_answer()` in answer_generation.py.
        Returns the final answer string.
        """
        assert reranked_df is not None and not reranked_df.empty, "reranked_df is empty"
        self._ensure_model()
        assert self._tokenizer is not None and self._model is not None

        if print_chunks:
            print(f"\n🔍 Query: {query}\n")
            print(" Top Reranked Chunks:\n")
            for i, row in reranked_df.head(top_k).iterrows():
                preview = str(row.chunk_text).strip().replace("\n", " ")[:200]
                cos = float(row.cosine_score) if "cosine_score" in reranked_df.columns else float("nan")
                rrs = float(row.rerank_score) if "rerank_score" in reranked_df.columns else float("nan")
                print(f"Rank #{i+1}")
                print(f"Chunk ID   : {row.chunk_id}")
                if cos == cos:  # not NaN
                    print(f"Cosine Sim : {cos:.4f}")
                if rrs == rrs:  # not NaN
                    print(f"Rerank     : {rrs:.4f}")
                print(f"Preview    : {preview}...\n")

        top_chunks = reranked_df.head(top_k)["chunk_text"].astype(str).tolist()
        prompt = self._build_prompt(query, top_chunks)

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        answer = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1]
        if "[/INST]" in answer:
            answer = answer.split("[/INST]")[-1]

        final = answer.strip()
        print("Final Answer:\n")
        print(final)
        return final


# Optional local sanity check with a tiny dummy DF
if __name__ == "__main__":
    # Minimal demo without touching your existing files
    data = {
        "chunk_text": [
            "Refunds will be issued to the original payment method within 5–7 business days after inspection.",
            "Holiday returns must be made within 30 days of purchase unless marked Final Sale.",
            "Engraved or customized items are not eligible for return unless defective upon arrival.",
        ],
        "doc_name": ["policy.txt", "policy.txt", "policy.txt"],
        "chunk_id": [0, 1, 2],
        "cosine_score": [0.77, 0.70, 0.68],
        "rerank_score": [4.6, 3.2, 2.9],
    }
    df = pd.DataFrame(data)
    agent = AnswererAgent()
    agent.answer("can you explain original payment method for refund?", df, top_k=3)
