#  ReturnMind – Multi-Agent RAG System for Return Policy Q&A  

**ReturnMind** is a modular **multi-agent Retrieval-Augmented Generation (RAG)** system built to answer customer questions about return policies.  
It combines **dense retrieval**, **fine-tuned reranking**, and **LLM-based answer generation** in a **multi-agent architecture** – making it **scalable, explainable, and production-ready**.  

---

## Features  

- **Multi-Agent Pipeline** →  
  - `RetrievalAgent` → fetches candidate chunks from Milvus  
  - `RerankerAgent` → scores chunks using a fine-tuned cross-encoder (BGE + LoRA regression head)  
  - `AnswererAgent` → generates grounded answers using Mistral-7B or Phi-3-Mini  
  - `MainAgent` → orchestrates the entire flow with feedback logging  
- **Document Chunking** → Splits policies into overlapping, sentence-aware chunks (spaCy).  
- **Dense Retrieval** → Embeddings generated with **BGE-base-en v1.5**, stored in **Milvus** (HNSW, cosine similarity).  
- **Fine-Tuned Reranker** → Custom **LoRA-adapted BGE Reranker** trained with regression head (scores ∈ [0,5]) for improved chunk ranking.  
- **Answer Generation** → Uses **Mistral-7B-Instruct** (or Phi-3-Mini) to produce concise, policy-grounded answers.  
- **Feedback Loop** → Interactive 1–5 rating stored in `feedback_log.csv` for future reinforcement learning.  


##  Project Structure  

ReturnMind/
│── BaseDocs/                      # Raw policy documents (.txt)
│── chunk_creation.py               # Splits docs into 300–400 word chunks
│── generate_embeddings.py          # Generates normalized embeddings (BGE)
│── ingest_to_milvus.py             # Inserts embeddings into Milvus w/ HNSW index
│── fine_tune_bge_reranker_lora.py  # Fine-tunes reranker with LoRA + regression
│── rerank_with_finetuned_model.py  # Reranks retrieved chunks
│── answer_generation.py            # Generates answers (Mistral-7B / Phi-3)
│── Agents/
│    ├── retrieval_agent.py
│    ├── reranker_agent.py
│    ├── reranker_agent_finetuned.py
│    ├── answerer_agent.py
│    └── main_agent.py              # Orchestrator Agent
│── main.py                         # Simple end-to-end runner w/ feedback
│── feedback_log.csv                # User ratings log


##  Pipeline Overview  

```mermaid
flowchart LR
    A[ Documents] -->|Chunking| B[Chunks CSV]
    B -->|Embeddings| C[Milvus Vector DB]
    Q[ User Query] --> D[RetrievalAgent]
    D --> E[RerankerAgent (Fine-tuned BGE)]
    E --> F[AnswererAgent (Mistral/Phi-3)]
    F --> G[Final Answer]
    G --> H[Feedback Log]


##  Setup & Installation  

1. **Clone repo**  
   bash
   git clone https://github.com/<your-username>/ReturnMind.git
   cd ReturnMind


2. **Install dependencies**  
   bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   

3. **Start Milvus (standalone)**  
   bash
   ./standalone.bat   # or docker compose up

4. **Prepare data**  
   bash
   python chunk_creation.py
   python generate_embeddings.py
   python ingest_to_milvus.py
   

5. **Run pipeline**  
   bash
   python main_agent.py --query "Can I return engraved items?" --top_k 3
   


## Fine-Tuned Reranker  

We fine-tune **BAAI/bge-reranker-base** with **LoRA** adapters and a regression head on a `(query, chunk, score)` dataset.  
- Loss → **MSE**  
- Eval → **MSE, Spearman correlation, nDCG@3**  
- Scores normalized to **[0,1] → mapped to [0,5]**  

This significantly improves ranking quality compared to vanilla cosine retrieval.

---

## Example Run  

**Query:**  
> *“Can you explain original payment method for refund?”*  

**Top Reranked Chunks:**  
```
Rank #1 | Rerank 4.62 | "Refunds will be issued to the original payment method..."
Rank #2 | Rerank 3.25 | "Store credit will be offered if payment card is unavailable..."
Rank #3 | Rerank 2.91 | "Refund timelines vary between 5–7 days after inspection..."
```

**Final Answer:**  
> Refunds are issued back to your original payment method within 5–7 business days. If the card is unavailable, store credit will be provided.

---

## 🛠️ Roadmap  

- [x] Modular **multi-agent refactor**  
- [x] Feedback loop logging  
- [ ] Add **LangChain/LangGraph integration**  
- [ ] Hybrid retrieval (sparse + dense)  
- [ ] Conversational memory for multi-turn queries  
- [ ] Deployment via **FastAPI / Streamlit / Hugging Face Spaces**  

---

## Contributing  

Contributions are welcome! Open issues, suggest improvements, or try integrating new retrievers/rerankers.  

---

## License  

MIT License.  
