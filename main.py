from rerank_with_finetuned_model import get_top_chunks
from answer_generation import generate_answer

import csv
from datetime import datetime

def log_feedback(query, df_reranked, answer, feedback):
    timestamp = datetime.now().isoformat()
    top_chunks = " || ".join(df_reranked["chunk_text"].head(3).tolist())

    with open("feedback_log.csv", mode="a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, query, top_chunks, answer, feedback])


def run_pipeline(query: str, top_k: int = 3):
    print(f"\n Running ReturnMind RAG pipeline for query: \"{query}\"")

    # Step 1: Retrieve reranked top chunks (returns full DataFrame with scores)
    df_reranked = get_top_chunks(query=query, top_k=top_k)

    # Step 2: Generate answer with Mistral
    answer = generate_answer(query, df_reranked, top_k=top_k)

    # Step 3: Final confirmation
    print("\n ---------------------Final Answer---------------------:\n")
    print(f"👉 {repr(answer)}")

    # Step 4: Ask for feedback
    print("\n Rate this answer from 1 to 5:")
    print("1 = Poor • 2 = Fair • 3 = Good • 4 = Very Good • 5 = Excellent")
    while True:
        feedback = input("Your rating: ").strip()
        if feedback in {"1", "2", "3", "4", "5"}:
            break
        else:
            print(" Please enter a number from 1 to 5.")
    
    # Step 5: Log feedback
    log_feedback(query, df_reranked, answer, feedback)
    print(" Feedback saved. Thank you!")


if __name__ == "__main__":
    # You can change this test query or make it dynamic later
    query = "can you explain original payment method for refund?"
    run_pipeline(query)
































#----------------- Code below is before adding Feedback logic

# from rerank_with_finetuned_model import get_top_chunks
# from answer_generation import generate_answer

# def run_pipeline(query: str, top_k: int = 3):
#     print(f"\n Running ReturnMind RAG pipeline for query: \"{query}\"")

#     # Step 1: Retrieve reranked top chunks (returns full DataFrame with scores)
#     df_reranked = get_top_chunks(query=query, top_k=top_k)

#     # Step 2: Generate answer with Mistral and show reranked metadata
#     answer = generate_answer(query, df_reranked, top_k=top_k)

#     # Step 3: Final confirmation (already printed inside g enerate_answer)
#     print("\n Final Answer:\n")
#     print(f"👉 {repr(answer)}")  # Shows any whitespace or formatting issues clearly

# if __name__ == "__main__":
#     #  Test your query here
#     query = "can you explain original payment method for refund?"
#     run_pipeline(query)
