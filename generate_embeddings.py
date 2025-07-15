import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# -----------------------------
# Load the chunked CSV
# -----------------------------
df = pd.read_csv("chunked_documents.csv")  # Ensure this file is in your project folder

# -----------------------------
# Load E5-base-v2 model + tokenizer
# -----------------------------
model_name = "BAAI/bge-base-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -----------------------------
# Embedding function
# -----------------------------
# def embed_text(text):
#     prompt = "passage: " + text  # E5 expects this prompt format
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
#     with torch.no_grad():
#         output = model(**inputs)
#     embedding = output.last_hidden_state[:, 0]  # Use the CLS-style token
#     return embedding.squeeze().cpu().numpy()

def embed_text(text):
    prompt = "passage: " + text  # E5 expects this prompt format
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        output = model(**inputs)
    
    # Get CLS-style token (first token of last hidden state)
    embedding = output.last_hidden_state[:, 0]  # shape: (1, hidden_size)
    
    # 🔹 Normalize the vector to unit length (L2 norm) for cosine similarity
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

    return embedding.squeeze().cpu().numpy()

# -----------------------------
# Embed all chunks
# -----------------------------
embeddings = []
print("Generating embeddings for all chunks...\n")
for chunk in tqdm(df["chunk_text"], desc="Embedding"):
    vector = embed_text(chunk)
    embeddings.append(vector)

# Add embeddings to DataFrame
df["embedding"] = embeddings

# Optionally, save for use in Milvus later
df.to_pickle("chunk_embeddings_e5.pkl")

print("\n Embeddings generated and saved to 'chunk_embeddings_e5.pkl'")
