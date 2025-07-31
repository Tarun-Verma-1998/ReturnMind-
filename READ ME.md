# 🧠 ReturnMind: AI-Powered Return Policy Assistant

> A modular, production-grade Retrieval-Augmented Generation (RAG) pipeline using Milvus, LoRA-finetuned reranker, and Mistral-7B to answer real-world customer return queries.

---

## 📌 What is ReturnMind?

**ReturnMind** is a custom-built RAG assistant that can answer customer questions from a company’s internal **return policy documents**. It mimics how a customer support agent would search through policy documents and provide a precise, policy-aligned answer.

It includes:

- **Retrieval using BGE embeddings** stored in **Milvus vector DB**
- **Reranking using a fine-tuned BGE-Reranker model (LoRA + regression head)**
- **Final answer generation using Mistral-7B-Instruct**

---

## 🧠 Architecture (High-Level Flow)

```text
+------------------+          +---------------------+        +-----------------------+
|   User Query     |   --->   |   BGE Query Embed   |  --->  |  Top-K Milvus Search  |
+------------------+          +---------------------+        +-----------------------+
                                                            |
                                                            v
                                                  +------------------------+
                                                  |  BGE-Reranker (LoRA)   |
                                                  |  Fine-tuned Regressor  |
                                                  +------------------------+
                                                            |
                                                            v
                                               +------------------------------+
                                               | Top-N Ranked Chunks (Text)   |
                                               +------------------------------+
                                                            |
                                                            v
                                         +----------------------------------------+
                                         | Mistral-7B-Instruct Answer Generator   |
                                         +----------------------------------------+
```

---

## 🔍 Core Agents and Intelligence Modules

The pipeline is structured around three intelligent agents that mimic how a human support rep would read, assess, and respond to a customer's query.

**1. Retriever Agent**: This module encodes the incoming query using a BGE (E5-style) encoder and searches through a Milvus vector database of precomputed chunk embeddings. It efficiently pulls out the top 30 most semantically relevant text chunks using cosine similarity. This mimics how a human would look for "relevant paragraphs" inside a document.

**2. Reranker Agent**: Instead of using cosine scores alone, the reranker is a LoRA-finetuned BGE-Reranker model trained on (query, chunk, relevance) triplets. It scores how likely each chunk is to actually answer the query. This step improves quality by promoting more "answerable" chunks to the top. The reranker uses a regression head (sigmoid + MSE) to output a normalized relevance score between 0 and 1.

**3. Answer Generation Agent**: The top reranked chunks are passed into a multi-passage prompt template, which is sent to Mistral-7B-Instruct. This open-weight LLM is instructed to (a) read the chunks, (b) choose the best one, and (c) return a concise, customer-friendly answer. If no chunk is relevant, it can gracefully return a fallback message.

---

## 🚀 Run Instructions

```bash
# 1. Chunk documents
python chunk_creation.py

# 2. Generate embeddings
python generate_embeddings.py

# 3. Ingest into Milvus
python ingest_to_milvus.py

# 4. (Optional) Fine-tune reranker
python fine_tune_bge_reranker_lora.py

# 5. Run full pipeline
python main.py
```

---

## 🔎 Sample Output

```bash
Query: Can I return customized or engraved items?

📊 Top Reranked Chunks:
Chunk ID: 4 | Cosine: 0.82 | Rerank: 0.91
Preview: Customized or engraved items may not be eligible for return unless...

💬 Final Answer:
Customized or engraved items can only be returned if damaged or incorrect. Otherwise, they are final sale.
```

---

## 💡 Key Highlights

- 🔹 **BGE (E5-style) Embeddings** → high-quality semantic retrieval
- 🔹 **Milvus Vector DB** → real-time similarity search
- 🔹 **LoRA Fine-tuning** → lightweight reranker adapted to custom domain
- 🔹 **Mistral-7B** → compact open-weight model for fast inference
- 🔹 **Modular Design** → each step can be replaced/swapped

---

## 🔄 Future Enhancements

- ✅ Add FastAPI wrapper for interactive querying
- ✅ Add answer confidence + explainability
- 🔲 Model versioning with MLflow
- 🔲 Streamlit or LangChain UI
- 🔲 Add evaluation set for NDCG / rerank lift

---

## 👤 Author

**Tarun Verma**\
ML Engineer | MLOps | LLM Architectures\
[LinkedIn](https://linkedin.com/in/your-profile)

---

## 📄 License

MIT License. Use freely with credit.

