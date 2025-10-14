# **ReturnMind — Multi-Agent RAG System for E-Commerce Return Policies**

> **A production-ready Retrieval-Augmented Generation (RAG) pipeline that answers customer queries about return/refund policies — powered by fine-tuned reranking and multi-agent orchestration.**

---

##  Overview

**ReturnMind** is an **LLM-powered multi-agent system** that retrieves, reranks, and generates grounded answers to user questions about company return policies.

The system mirrors **real-world search and recommendation pipelines**, integrating:

- **Dense retrieval** (Milvus + BGE embeddings)
- **Fine-tuned reranking** (BGE-Reranker with LoRA regression head)
- **Answer generation** (Mistral-7B-Instruct or Phi-3-Mini)
- **Feedback logging** for continual improvement
- **Agent-based modular design** for scalability and extensibility

---

##  Architecture

```
          ┌───────────────────────────────┐
          │         User Query            │
          └───────────────────────────────┘
                         │
                         ▼
              ┌───────────────────┐
              │  Retrieval Agent  │
              │  (Milvus + BGE)   │
              └───────────────────┘
                         │ top 30 chunks
                         ▼
              ┌───────────────────┐
              │  Reranker Agent   │
              │ (Fine-tuned BGE   │
              │   w/ LoRA head)   │
              └───────────────────┘
                         │ top 3 chunks
                         ▼
              ┌───────────────────┐
              │  Answerer Agent   │
              │ (Mistral-7B or    │
              │   Phi-3-Mini)     │
              └───────────────────┘
                         │
                         ▼
          ┌───────────────────────────────┐
          │  Final Grounded Answer        │
          │  + Optional Feedback Logging  │
          └───────────────────────────────┘
```

---

##  Core Pipeline

### 1️ **Document Chunking** — `chunk_creation.py`
- Splits base documents (`BaseDocs/*.txt`) into 400-word chunks using spaCy sentence segmentation.
- Overlapping window ensures context continuity.
- Outputs: `chunked_documents.csv`

### 2️ **Embedding Generation** — `generate_embeddings.py`
- Embeds chunks using **BAAI/bge-base-en-v1.5** (E5-style).
- Normalizes embeddings for **cosine similarity** search.
- Outputs: `chunk_embeddings_e5.pkl`

### 3️ **Vector Storage & Indexing** — `ingest_to_milvus.py`
- Inserts embeddings into **Milvus** with **HNSW index (M=16, efConstruction=200)**.
- Metric: **COSINE**  
- Collection name: `chunk_embeddings`

### 4️ **Reranker Fine-Tuning** — `fine_tune_bge_reranker_lora.py`
- Fine-tunes **BGE-Reranker-Base** using **LoRA adapters** and a **regression head**.
- Task: predict similarity scores (0–5) between queries and chunks.
- Evaluated with **MSE**, **Spearman correlation**, and **nDCG@3**.
- Best model saved under: `bge_reranker_lora_sigmoid_best/`

### 5 **Reranking Agent** — `reranker_agent_finetuned.py`
- Loads LoRA-adapted reranker with saved regression head.
- Computes **sigmoid(·) × 5** relevance scores for (query, passage) pairs.
- Returns top-ranked chunks for final generation.

### 6️ **Answer Generation** — `answer_generation.py` / `answerer_agent.py`
- Uses **Mistral-7B-Instruct** *(or lightweight Phi-3-Mini)* to craft concise, grounded answers.
- Context: top-k reranked chunks.
- Output: 1–2 sentence answer covering key return conditions.

### 7️ **Pipeline Orchestration** — `main_agent.py`
- Orchestrates:
  ```
  RetrievalAgent → RerankerAgentFineTuned → AnswererAgent
  ```
- Optional user feedback (`1–5`) logged to `feedback_log.csv`.
- CLI interface for easy testing.

---

##  Multi-Agent Design

| **Agent** | **Role** | **Core File** |
|------------|-----------|----------------|
|  Retrieval Agent | Fetches top-K candidates from Milvus (HNSW) | `retrieval_agent.py` |
|  Reranker Agent | Scores each candidate using fine-tuned BGE cross-encoder | `reranker_agent_finetuned.py` |
|  Answerer Agent | Generates final concise answers grounded in top chunks | `answerer_agent.py` |
|  Main Agent | Orchestrates all sub-agents and handles feedback loop | `main_agent.py` |

---

##  Tech Stack

| Category | Tools / Models |
|-----------|----------------|
| **Embeddings** | `BAAI/bge-base-en-v1.5` |
| **Vector DB** | Milvus (HNSW index, COSINE metric) |
| **Reranker** | `BAAI/bge-reranker-base` + LoRA fine-tuning |
| **LLM for Answering** | `mistralai/Mistral-7B-Instruct-v0.2` or `microsoft/Phi-3-Mini-4K-Instruct` |
| **Training Framework** | PyTorch, PEFT (LoRA), Transformers |
| **Pipeline Agents** | Modular classes for retrieval, reranking, and answering |
| **Monitoring** | Feedback logs (`feedback_log.csv`) for future retraining |
| **Environment** | Docker + Milvus + CUDA-enabled GPU |

---

##  Run Locally

### 1️ Start Milvus (Standalone)
```bash
docker pull milvusdb/milvus:v2.4.0
bash standalone.bat   # or ./standalone.sh
```

### 2️ Prepare Documents
Put your `.txt` files in `BaseDocs/`, then run:
```bash
python chunk_creation.py
python generate_embeddings.py
python ingest_to_milvus.py
```

### 3️ (Optional) Fine-Tune the Reranker
```bash
python fine_tune_bge_reranker_lora.py
```

### 4️ Run Full Pipeline
```bash
python main_agent.py --query "Can I return engraved items?" --top_k 3
```

You’ll see:
- Retrieved + reranked chunks
- Generated answer
- Option to rate response (1–5)

---

##  Example Output

**Query:**  
> *“Can you explain original payment method for refund?”*

**Top Reranked Chunks:**  
1. Refunds are issued to the original payment method within 5–7 days.  
2. Store credit applies only when the original payment method is unavailable.  
3. Cash refunds are not supported for online purchases.

**Final Answer:**  
>  *Refunds are processed back to your original payment method within 5–7 business days after inspection.*

**Feedback:**  
>  Rated “5 – Excellent” → Logged in `feedback_log.csv`

---

##  Why This Project Matters

- Demonstrates **end-to-end RAG orchestration** with real models (BGE, Mistral).
- Implements **retrieval + reranking + generation** just like industrial **search/recommendation** systems.
- Uses **fine-tuned LoRA regression reranker**, not just out-of-the-box scoring.
- Provides a **scalable, modular agent architecture** ready for LangChain / LangGraph integration.
- Collects user feedback for **continual model improvement**.

---

##  Future Enhancements

-  **Feedback-based retraining loop**
-  **LangGraph orchestration** for parallel agent reasoning
-  **Hybrid search** (BM25 + Vector)
-  **Conversational memory** and follow-up query handling
-  **Streamlit / FastAPI deployment**
-  **Prometheus / Evidently AI monitoring**

---

## 👨 Author

**Tarun Verma**  
Machine Learning Engineer — specialized in real-time RAG, ranking, and LLM deployment pipelines.  
 Toronto / Windsor, Canada  
 [tarunverma.ml@gmail.com](mailto:tarunverma.ml@gmail.com)  
 [LinkedIn](https://linkedin.com/in/tarunv-ai) | [GitHub](https://github.com/tarunv-ai)
