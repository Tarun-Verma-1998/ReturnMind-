from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer
import pandas as pd

# Step 1: Connect to Milvus
connections.connect(alias="default", host="localhost", port="19530")

# Step 2: Load your collection
collection = Collection(name="chunk_embeddings")

# Step 3: Load the same model used for document embeddings
model = SentenceTransformer("intfloat/e5-base-v2")

# Step 4: Embed the user query
def embed_query(query_text):
    return model.encode(f"query: {query_text}", normalize_embeddings=True).tolist()

# Step 5: Search for top-k similar chunks
def search_top_k(query, k=5):
    embedding = embed_query(query)
    results = collection.search(
        data=[embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 200}},
        limit=k,
        output_fields=["doc_name", "chunk_id", "chunk_text"]
    )
    return results[0]

# Step 6: Run an example search
if __name__ == "__main__":
    user_query = "How do I return an item if it's damaged?"
    top_hits = search_top_k(user_query, k=5)
    print(" ")
    print("QUERY: " ,user_query)
    print(" ")
    print("Top retrieved chunks from MILVUS are: ")

    for i, hit in enumerate(top_hits, 1):
        print(f"\nRank #{i}")
        print(f"Doc: {hit.entity.get('doc_name')}")
        print(f"Chunk ID: {hit.entity.get('chunk_id')}")
        print(f"Score: {hit.distance:.4f}")
        print(f"Text:\n{hit.entity.get('chunk_text')[:400]}...\n")
