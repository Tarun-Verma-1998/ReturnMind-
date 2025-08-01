# Add this to rerank_with_finetuned_model.py

import torch
import pandas as pd
from pymilvus import connections, Collection
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
E5_MODEL = "BAAI/bge-base-en-v1.5"
BGE_MODEL_DIR = "bge_reranker_lora_sigmoid_best"
COLLECTION_NAME = "chunk_embeddings"

def get_top_chunks(query: str, top_k: int = 3):
    # ----- Connect to Milvus -----
    connections.connect(host="localhost", port="19530")
    collection = Collection(COLLECTION_NAME)
    collection.load()

    # ----- Embed Query -----
    from transformers import AutoModel as E5Model, AutoTokenizer as E5Tokenizer
    e5_model = E5Model.from_pretrained(E5_MODEL).to(DEVICE)
    e5_tokenizer = E5Tokenizer.from_pretrained(E5_MODEL)
    with torch.no_grad():
        tokens = e5_tokenizer(query, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        output = e5_model(**tokens)
        query_embedding = output.last_hidden_state[:, 0, :]
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1).cpu().numpy()

    # ----- Search Milvus -----
    search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
    results = collection.search(
        data=query_embedding,
        anns_field="embedding",
        param=search_params,
        limit=top_k * 10,  # fetch more for reranker to choose best
        output_fields=["chunk_text", "doc_name", "chunk_id"]
    )

    retrieved = []
    for hit in results[0]:
        retrieved.append({
            "chunk_text": hit.entity.get("chunk_text"),
            "doc_name": hit.entity.get("doc_name"),
            "chunk_id": hit.entity.get("chunk_id"),
            "cosine_score": hit.distance
        })

    df_retrieved = pd.DataFrame(retrieved)

    # ----- Load Fine-tuned Reranker -----
    class RerankerRegressor(nn.Module):
        def __init__(self, model_name):
            super().__init__()
            self.base = AutoModel.from_pretrained(model_name)
            self.regression = nn.Linear(self.base.config.hidden_size, 1)
            self.activation = nn.Sigmoid()

        def forward(self, input_ids, attention_mask):
            outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            score = self.regression(cls_embedding)
            return self.activation(score).squeeze()

    tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL_DIR)
    model = RerankerRegressor(BGE_MODEL_DIR).to(DEVICE)
    model.eval()

    # ----- Rerank -----
    rerank_data = []
    for row in df_retrieved.itertuples():
        text = f"Query: {query} Document: {row.chunk_text}"
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            score = model(**encoded).item()
        rerank_data.append({
            "chunk_text": row.chunk_text,
            "doc_name": row.doc_name,
            "chunk_id": row.chunk_id,
            "cosine_score": row.cosine_score,
            "rerank_score": round(score, 3)
        })

    df_reranked = pd.DataFrame(rerank_data)
    df_reranked = df_reranked.sort_values(by="rerank_score", ascending=False)
    df_reranked.to_csv("reranked_results.csv", index=False)  # Save output

    return df_reranked





















































































#-------------------------------------------------------------------------------------------------------------------------------------
#THIS CODE IS FOR EXPERIMENT (WITHOUT FUNCTION). MAIN CODE IS IS THE TOP | 




# import torch
# import pandas as pd
# from pymilvus import connections, Collection
# from transformers import AutoTokenizer, AutoModel
# from peft import PeftModel
# from torch import nn

# # ----- CONFIG -----
# MILVUS_HOST = "localhost"
# MILVUS_PORT = "19530"
# COLLECTION_NAME = "chunk_embeddings"
# TOP_K = 30
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# # QUERY = "How do I return an item if it's damaged?"
# # QUERY = "What are the return windows for electronics and furniture items?"
# # QUERY = "Can I return a used home appliance?"
# # QUERY = "can i return on holidays ?"
# # QUERY = "Can I return items bought during holidays?"
# # QUERY = "whats about to return engraved product?"
# # QUERY = "Can I return customized or engraved items?" ------------------------------------------------------ Good one
# QUERY = "Can I return items bought during holiday sales?"
# E5_MODEL = "BAAI/bge-base-en-v1.5"
# BGE_MODEL_DIR = "bge_reranker_lora_sigmoid_best"


# # ----- CONNECT TO MILVUS -----
# connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
# collection = Collection(COLLECTION_NAME)

# # ----- Embed Query Using E5 -----
# from transformers import AutoModel as E5Model, AutoTokenizer as E5Tokenizer
# e5_model = E5Model.from_pretrained(E5_MODEL).to(DEVICE)
# e5_tokenizer = E5Tokenizer.from_pretrained(E5_MODEL)

# with torch.no_grad():
#     tokens = e5_tokenizer(QUERY, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
#     output = e5_model(**tokens)
#     query_embedding = output.last_hidden_state[:, 0, :]
#     query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1).cpu().numpy()

# # ----- Search Milvus -----
# collection.load()
# search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
# results = collection.search(
#     data=query_embedding,
#     anns_field="embedding",
#     param=search_params,
#     limit=TOP_K,
#     output_fields=["chunk_text", "doc_name", "chunk_id"]
# )

# retrieved = []
# for hit in results[0]:
#     retrieved.append({
#         "chunk_text": hit.entity.get("chunk_text"),
#         "doc_name": hit.entity.get("doc_name"),
#         "chunk_id": hit.entity.get("chunk_id"),
#         "cosine_score": hit.distance
#     })

# df_retrieved = pd.DataFrame(retrieved)
# print("\nDEBUG: Retrieved Document Names + Chunk IDs:")
# for row in df_retrieved.itertuples():
#     print(f"- {row.doc_name}, Chunk ID: {row.chunk_id}")


# # ----- Load Fine-tuned BGE-Reranker + LoRA -----
# class RerankerRegressor(nn.Module):
#     def __init__(self, model_name):
#         super().__init__()
#         self.base = AutoModel.from_pretrained(model_name)
#         self.regression = nn.Linear(self.base.config.hidden_size, 1)
#         self.activation = nn.Sigmoid()  # <- this line is crucial

#     def forward(self, input_ids, attention_mask):
#         outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
#         cls_embedding = outputs.last_hidden_state[:, 0, :]
#         score = self.regression(cls_embedding)
#         return self.activation(score).squeeze()  # <- squash between 0 and 1

# tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL_DIR)
# model = RerankerRegressor(BGE_MODEL_DIR).to(DEVICE)
# model.eval()

# # ----- Rerank Using Fine-tuned Model -----
# rerank_data = []
# for row in df_retrieved.itertuples():
#     text = f"Query: {QUERY} Document: {row.chunk_text}"
#     encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
#     with torch.no_grad():
#         score = model(**encoded).item()
#     rerank_data.append({
#         "chunk_text": row.chunk_text,
#         "doc_name": row.doc_name,
#         "chunk_id": row.chunk_id,
#         "cosine_score": row.cosine_score,
#         "rerank_score": round(score, 3)
#     })

# df_reranked = pd.DataFrame(rerank_data)
# df_reranked = df_reranked.sort_values(by="rerank_score", ascending=False)

# # ----- Output -----
# print("\nTop reranked results:")
# for i, row in df_reranked.iterrows():
#     print(f"\nRank #{i+1}")
#     print(f"Doc: {row['doc_name']}, Chunk ID: {row['chunk_id']}")
#     print(f"Score: {row['rerank_score']}")
#     print(f"Text: {row['chunk_text'][:300]}...")

# # Optional: Save to CSV
# df_reranked.to_csv("reranked_results.csv", index=False)
