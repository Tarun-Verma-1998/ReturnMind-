"""
Fine-tune BGE Reranker with LoRA + Regression Head (Sigmoid)
-------------------------------------------------------------------------------
- Base model: BAAI/bge-reranker-base (cross-encoder)
- Adds a small regression head on top of CLS embedding and trains it
- Applies LoRA adapters to the base encoder (query/key/value)
- Saves BOTH: (1) LoRA adapters + tokenizer to BEST_MODEL_DIR, and
              (2) regression head weights to BEST_MODEL_DIR/regression_head.pt
- Evaluates with MSE, Spearman, and nDCG@3

This script is a DROP-IN replacement for your previous fine-tuner, with the fix
that the trained regression head is now saved.
"""
from __future__ import annotations

import os
import json
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score


# -----------------------------
# Configuration
# -----------------------------
@dataclass
class Config:
    MODEL_NAME: str = "BAAI/bge-reranker-base"
    CSV_PATH: str = "bge_reranker_training_data.csv"  # expects columns: query, chunk_text, score (0..5)
    BEST_MODEL_DIR: str = "bge_reranker_Final"

    MAX_LEN: int = 512 
    BATCH_SIZE: int = 8
    EPOCHS: int = 5
    LR: float = 2e-5
    WEIGHT_DECAY: float = 0.0
    WARMUP_RATIO: float = 0.06

    PATIENCE: int = 2
    MIN_DELTA: float = 1e-3

    SEED: int = 42


CFG = Config()


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Dataset
# -----------------------------
class RerankerDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: AutoTokenizer, max_len: int):
        self.df = dataframe.reset_index(drop=True)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        text = f"Query: {row['query']} Document: {row['chunk_text']}"
        enc = self.tok(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        # Normalize target to [0, 1]
        score = torch.tensor(float(row["score"]) / 5.0, dtype=torch.float32)
        return input_ids, attention_mask, score


# -----------------------------
# Model
# -----------------------------
class RerankerRegressor(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.base = AutoModel.from_pretrained(model_name)
        self.regression = nn.Linear(self.base.config.hidden_size, 1)
        self.activation = nn.Sigmoid()  # map to [0, 1]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        score = self.regression(cls)
        return self.activation(score).squeeze(-1)


# -----------------------------
# Utils
# -----------------------------
def count_trainable_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_best(model: RerankerRegressor, tokenizer: AutoTokenizer, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # Save LoRA-adapted base encoder + tokenizer
    model.base.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    # Save the trained regression head weights
    torch.save(model.regression.state_dict(), os.path.join(out_dir, "regression_head.pt"))
    # Minimal metadata for clarity
    meta = {
        "model_name": CFG.MODEL_NAME,
        "max_len": CFG.MAX_LEN,
        "head": {
            "in_features": model.base.config.hidden_size,
            "out_features": 1,
            "activation": "Sigmoid",
            "target_scale": "0..1 (multiply by 5 at inference if you want 0..5)",
        },
    }
    with open(os.path.join(out_dir, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    set_seed(CFG.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    df = pd.read_csv(CFG.CSV_PATH)
    assert {"query", "chunk_text", "score"}.issubset(df.columns), (
        "CSV must contain columns: query, chunk_text, score"
    )

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=CFG.SEED)
    train_df.to_csv("bge_reranker_train.csv", index=False)
    val_df.to_csv("bge_reranker_val.csv", index=False)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME)

    # Datasets / Loaders
    train_ds = RerankerDataset(train_df, tokenizer, CFG.MAX_LEN)
    val_ds = RerankerDataset(val_df, tokenizer, CFG.MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, drop_last=False)

    # Model + LoRA on base encoder
    model = RerankerRegressor(CFG.MODEL_NAME)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "key", "value"],  # keep consistent with your previous setup
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model.base = get_peft_model(model.base, peft_config)

    # Freeze everything except LoRA params and the regression head
    for name, p in model.named_parameters():
        # LoRA-injected params have requires_grad=True already; keep head trainable
        if "lora_" in name or name.startswith("regression"):
            p.requires_grad = True
        else:
            # In PEFT, base weights are typically frozen; ensure that here explicitly
            p.requires_grad = p.requires_grad and ("lora_" in name or name.startswith("regression"))

    model.to(device)

    total, trainable = count_trainable_parameters(model)
    print(f"Total params: {total:,} | Trainable: {trainable:,}")

    # Optimizer & Scheduler
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)

    num_training_steps = CFG.EPOCHS * len(train_loader)
    num_warmup_steps = int(CFG.WARMUP_RATIO * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    loss_fn = nn.MSELoss()

    # Early stopping on Val MSE
    best_val_mse = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, CFG.EPOCHS + 1):
        # ----------------- Train -----------------
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CFG.EPOCHS} - Training")
        for input_ids, attention_mask, labels in pbar:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            preds = model(input_ids, attention_mask)  # [batch]
            loss = loss_fn(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * input_ids.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train = train_loss / len(train_ds)

        # ----------------- Validate -----------------
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for input_ids, attention_mask, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{CFG.EPOCHS} - Validation"):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                preds = model(input_ids, attention_mask)
                val_preds.extend(preds.detach().cpu().numpy().tolist())
                val_labels.extend(labels.detach().cpu().numpy().tolist())

        # Metrics (labels/preds are in [0,1])
        val_mse = mean_squared_error(val_labels, val_preds)
        spearman_corr, _ = spearmanr(val_labels, val_preds)
        # For nDCG, wrap as single row (flat list)
        ndcg3 = ndcg_score([val_labels], [val_preds], k=3)

        print(
            f"Epoch {epoch}: TrainLoss={avg_train:.4f} | ValMSE={val_mse:.4f} | "
            f"Spearman={spearman_corr:.4f} | nDCG@3={ndcg3:.4f}"
        )

        # ----------------- Early Stopping + Save -----------------
        if best_val_mse - val_mse > CFG.MIN_DELTA:
            best_val_mse = val_mse
            epochs_no_improve = 0
            save_best(model, tokenizer, CFG.BEST_MODEL_DIR)
            print(f"  ✅ New best saved to: {CFG.BEST_MODEL_DIR} (ValMSE={val_mse:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s)")
            if epochs_no_improve >= CFG.PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Val MSE: {best_val_mse:.4f}")
    print("Saved files:")
    print(f" - {os.path.join(CFG.BEST_MODEL_DIR, 'adapter_model.safetensors')} (LoRA)")
    print(f" - {os.path.join(CFG.BEST_MODEL_DIR, 'adapter_config.json')} (LoRA config)")
    print(f" - {os.path.join(CFG.BEST_MODEL_DIR, 'tokenizer_config.json')}")
    print(f" - {os.path.join(CFG.BEST_MODEL_DIR, 'regression_head.pt')} (REGRESSION HEAD)\n")


if __name__ == "__main__":
    main()
