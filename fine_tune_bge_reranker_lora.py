import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import AdamW

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# EV For evaluation (spearman and ndcg score)
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

from tqdm import tqdm
import os

# ---------- Configuration ----------
MODEL_NAME = "BAAI/bge-reranker-base"
CSV_PATH = "bge_reranker_training_data.csv"
MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 5
LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PATIENCE = 2
MIN_DELTA = 0.001
BEST_MODEL_DIR = "bge_reranker_lora_sigmoid_best"

# ---------- Dataset Class ----------
class RerankerDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = f"Query: {row['query']} Document: {row['chunk_text']}"
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()
        score = torch.tensor(float(row["score"]) / 5.0, dtype=torch.float)  # normalize between 0 and 1
        return input_ids, attention_mask, score

# ---------- Load & Split Data ----------
df = pd.read_csv(CSV_PATH)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_csv("bge_reranker_train.csv", index=False)
val_df.to_csv("bge_reranker_val.csv", index=False)

# ---------- Tokenizer ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_dataset = RerankerDataset(train_df, tokenizer, MAX_LEN)
val_dataset = RerankerDataset(val_df, tokenizer, MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ---------- Model with Sigmoid Activation ----------
class RerankerRegressor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.base = AutoModel.from_pretrained(model_name)
        self.regression = nn.Linear(self.base.config.hidden_size, 1)
        self.activation = nn.Sigmoid()  # squash between 0 and 1

    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        score = self.regression(cls_embedding)
        return self.activation(score).squeeze()

# ---------- Initialize Model and LoRA ----------
model = RerankerRegressor(MODEL_NAME)
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "key", "value"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION
)
model.base = get_peft_model(model.base, peft_config)
model.to(DEVICE)

# ---------- Optimizer & Loss ----------
optimizer = AdamW(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# ---------- Training with Early Stopping ----------
best_val_mse = float("inf")
epochs_no_improve = 0

for epoch in range(EPOCHS):
    total_train_loss = 0
    model.train()
    for input_ids, attention_mask, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} - Training"):
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    # ---------- Validation ----------
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} - Validation"):
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(input_ids, attention_mask)
            val_preds.extend(outputs.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    # val_mse = mean_squared_error(val_labels, val_preds)
    # print(f"Epoch {epoch+1}: Train Loss = {total_train_loss:.4f}, Val MSE = {val_mse:.4f}")

    # EV Evaluation Metrics (Updated)

        # --- Compute MSE ---
    val_mse = mean_squared_error(val_labels, val_preds) 

    # --- Compute Spearman Correlation ---
    spearman_corr, _ = spearmanr(val_labels, val_preds)

    # --- Compute nDCG@3 ---
    # sklearn expects a 2D array: shape (n_queries, n_chunks_per_query)
    # But we only have one flat list, so we wrap them as single row
    true_relevance = [val_labels]
    predicted_scores = [val_preds]
    ndcg_3 = ndcg_score(true_relevance, predicted_scores, k=3)

    # --- Print All ---
    print(f"Epoch {epoch+1}: Train Loss = {total_train_loss:.4f}, Val MSE = {val_mse:.4f}, Spearman = {spearman_corr:.4f}, nDCG@3 = {ndcg_3:.4f}")


    # ---------- Early Stopping Check ----------
    if best_val_mse - val_mse > MIN_DELTA:
        best_val_mse = val_mse
        epochs_no_improve = 0
        # Save best model
        os.makedirs(BEST_MODEL_DIR, exist_ok=True)
        model.base.save_pretrained(BEST_MODEL_DIR)
        tokenizer.save_pretrained(BEST_MODEL_DIR)
        print(f" New best model saved (Val MSE = {val_mse:.4f})")
    else:
        epochs_no_improve += 1
        print(f" No improvement for {epochs_no_improve} epoch(s)")

    if epochs_no_improve >= PATIENCE:
        print("Early stopping triggered")
        break

print(f"Training complete. Best Val MSE: {best_val_mse:.4f}")



