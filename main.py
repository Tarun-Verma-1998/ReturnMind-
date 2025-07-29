from rerank_with_finetuned_model import get_top_chunks
from answer_generation import generate_answer

def run_pipeline(query: str, top_k: int = 3):
    print(f"\n🔍 Running ReturnMind RAG pipeline for query: '{query}'")

    # Step 1: Rerank & Retrieve Top Chunks
    top_chunks = get_top_chunks(query=query, top_k=top_k)
    print("\n📘 Top Reranked Chunks:")
    for i, chunk in enumerate(top_chunks, 1):
        print(f"\nChunk #{i}:\n{chunk[:300]}...\n")  # show first 300 chars

    # Step 2: Generate Answer using Mistral
    answer = generate_answer(query, top_chunks)

    # Step 3: Show Final Answer
    print("\n Final Answer:\n")
    print(answer)


if __name__ == "__main__":
    # Try with any query here
    query = "Can I return items bought during holiday sales?"
    run_pipeline(query)