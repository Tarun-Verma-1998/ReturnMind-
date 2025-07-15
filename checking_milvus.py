from pymilvus import connections, Collection

# Connect to Milvus
connections.connect(alias="default", host="localhost", port="19530")

# Load the collection
collection = Collection("chunk_embeddings")
collection.load()

print(" Total records:", collection.num_entities)

results = collection.query(
    expr="chunk_id >= 0",  # you can change this filter
    output_fields=["chunk_id", "doc_name", "chunk_text", "embedding"],
    limit=3
)

# Pretty print
for i, res in enumerate(results):
    print(f"\n🔹 Record {i+1}:")
    print(f"Chunk ID: {res['chunk_id']}")
    print(f"Doc Name: {res['doc_name']}")
    print(f"Chunk Text: {res['chunk_text'][:200]}...")  # Truncated
    print(f"Embedding (first 5 dims): {res['embedding'][:5]} ...\n")
