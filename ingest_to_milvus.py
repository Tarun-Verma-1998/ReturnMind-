from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

# ----------------------------------------
# 1. Connect to Milvus
# ----------------------------------------
connections.connect(alias="default", host="localhost", port="19530")

# ----------------------------------------
# 2. Load embeddings from .pkl file
# ----------------------------------------
df = pd.read_pickle("chunk_embeddings_e5.pkl")  # Adjust path if needed

# ----------------------------------------
# 3. Define collection schema
# ----------------------------------------
fields = [
    FieldSchema(name="chunk_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="doc_name", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=2000),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
]
schema = CollectionSchema(fields, description="ReturnMind - E5 chunk embeddings")

# ----------------------------------------
# 4. Create or reset collection
# ----------------------------------------
collection_name = "chunk_embeddings"
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
collection = Collection(name=collection_name, schema=schema)

# ----------------------------------------
# 5. Normalize embeddings and prepare data
# ----------------------------------------
raw_embeddings = np.stack(df["embedding"].values)
normalized_embeddings = normalize(raw_embeddings, norm="l2")  # L2 normalization for cosine similarity

data_to_insert = [
    df["chunk_id"].tolist(),
    df["doc_name"].tolist(),
    df["chunk_text"].tolist(),
    normalized_embeddings
]

# ----------------------------------------
# 6. Insert and flush
# ----------------------------------------
collection.insert(data_to_insert)
collection.flush()

# ----------------------------------------
# 7. Create HNSW index on embedding field
# ----------------------------------------
index_params = {
    "index_type": "HNSW",
    "params": {
        "M": 16,
        "efConstruction": 200
    },
    "metric_type": "COSINE"
}
collection.create_index(field_name="embedding", index_params=index_params)

# ----------------------------------------
# 8. Load collection into memory
# ----------------------------------------
collection.load()

print("Embeddings successfully normalized, inserted, and indexed in Milvus.")
