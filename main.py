from rerank_with_finetuned_model import get_top_chunks
from answer_generation import generate_answer

def run_pipeline(query: str, top_k: int = 3):
    print(f"\n Running ReturnMind RAG pipeline for query: \"{query}\"")

    # Step 1: Retrieve reranked top chunks (returns full DataFrame with scores)
    df_reranked = get_top_chunks(query=query, top_k=top_k)

    # Step 2: Generate answer with Mistral and show reranked metadata
    answer = generate_answer(query, df_reranked, top_k=top_k)

    # Step 3: Final confirmation (already printed inside g enerate_answer)
    print(f"\n Answer generation complete.\n")

if __name__ == "__main__":
    #  Test your query here
    query = "How do I return an item if it's damaged?"
    run_pipeline(query)
