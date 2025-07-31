# answer_generation.py

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------- CONFIG ---------
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------- MAIN FUNCTION ---------
def generate_answer(query: str, reranked_df: pd.DataFrame, top_k: int = 3) -> str:
    """
    Uses top-k reranked chunks to generate answer using Mistral.
    Prints reranked metadata and final answer.
    """

    print(f"\n🔍 Query: {query}\n")

    # --------- STEP 1: Show reranked chunk metadata ---------
    print(" Top Reranked Chunks:\n")
    for i, row in reranked_df.head(top_k).iterrows():
        preview = row.chunk_text.strip().replace("\n", " ")[:200]
        print(f"Rank #{i+1}")
        print(f"Chunk ID   : {row.chunk_id}")
        print(f"Cosine Sim : {row.cosine_score:.4f}")
        print(f"Rerank     : {row.rerank_score:.4f}")
        print(f"Preview    : {preview}...\n")

    # --------- STEP 2: Prepare prompt for Mistral ---------
    top_chunks = reranked_df.head(top_k)["chunk_text"].tolist()
    context = "\n".join([f"- {chunk}" for chunk in top_chunks])
    prompt = f"""[INST] You are a helpful assistant answering customer queries using the company's return policy.
Context:
{context}

Question: {query}
Answer concisely and helpfully. [/INST]
"""

    # --------- STEP 3: Load Mistral model ---------
    print(" Loading Mistral-7B-Instruct model...\n")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        # offload_folder="offload"
    )
    model.eval()
    print(" Mistral model loaded.\n")

    # --------- STEP 4: Generate answer ---------
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1]
    if "[/INST]" in answer:
        answer = answer.split("[/INST]")[-1]

    # --------- STEP 5: Final Answer ---------
    print("Final Answer:\n")
    print(answer.strip())
    return answer.strip()



























#-------------------------------------------------------------------------------------------------------------------------------------
#THIS CODE IS FOR EXPERIMENT (WITHOUT FUNCTION). MAIN CODE IS IS THE TOP | 


# # answer_generation.py

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # --------- CONFIG ---------
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # --------- LOAD MODEL SAFELY ---------
# print("Loading Mistral-7B-Instruct model...")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     torch_dtype=torch.float16,
#     device_map="auto",  # let Hugging Face split between GPU/CPU
#     offload_folder="offload",
#     trust_remote_code=True  # safe fallback even if not strictly needed
# )
# model.eval()
# print(" Mistral model loaded.\n")

# # --------- MAIN FUNCTION ---------
# def generate_answer(query: str, top_chunks: list[str]) -> str:
#     """
#     Generates a final answer using Mistral LLM based on top-ranked chunks.

#     Args:
#         query (str): The user question.
#         top_chunks (List[str]): Top N chunk texts retrieved & reranked.

#     Returns:
#         str: Final answer string.
#     """
#     # Format context
#     context = "\n".join([f"- {chunk}" for chunk in top_chunks])
#     prompt = f"""[INST] You are a helpful assistant answering customer queries using the company's return policy.
# Context:
# {context}

# Question: {query}
# Answer concisely and helpfully. [/INST]
# """

#     # Tokenize and generate
#     inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=150,
#             temperature=0.7,
#             top_p=0.9,
#             do_sample=True,
#             repetition_penalty=1.1,
#             eos_token_id=tokenizer.eos_token_id
#         )

#     # Decode and clean
#     answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     # Debug print
#     print("\n RAW LLM OUTPUT:\n")
#     print(repr(answer))  # Show unprocessed output

#     # Clean up final string
#     if "Answer:" in answer:
#         answer = answer.split("Answer:")[-1]
#     if "[/INST]" in answer:
#         answer = answer.split("[/INST]")[-1]

#     return answer.strip()


# --------- Optional Standalone Test ---------
# if __name__ == "__main__":
#     test_query = "Can I return items bought during holiday sales?"
#     sample_chunks = [
#         "Final Sale or Clearance Items: Clearly marked on product pages. These are not eligible for return unless damaged.",
#         "If any of these items arrive damaged or incorrect, please contact us within 48 hours. We’ll issue store credit or send a replacement.",
#         "Holiday purchases may be marked final sale; please review product page notes before buying during promotions."
#     ]
#     final = generate_answer(test_query, sample_chunks)
#     print("\n Final Answer:\n")
    # print(final)


# #----------------------------------------------------------------Flan-t5-----------------------------------------------------------

# # import pandas as pd
# # from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# # import torch

# # # ----- Config -----
# # QUERY = "Can I return customized or engraved items?"
# # TOP_K = 3
# # MODEL_NAME = "google/flan-t5-base"
# # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # # ----- Load Reranked Chunks -----
# # df = pd.read_csv("reranked_results.csv")
# # top_chunks = df.sort_values("rerank_score", ascending=False).head(TOP_K)["chunk_text"].tolist()

# # # ----- Multi-shot Format -----
# # passages = "\n\n".join([f"Passage {i+1}: {chunk}" for i, chunk in enumerate(top_chunks)])
# # prompt = f"""
# # You are a helpful assistant at ReturnMind.

# # Below are some passages from ReturnMind's return policy.

# # Your task is to:
# # 1. Find the passage most relevant to the customer's question.
# # 2. Answer the question using only that passage.
# # 3. If the information is not present, say:
# #    "Sorry, I couldn't find this information in the policy."

# # {passages}

# # Customer Query: {QUERY}
# # Answer:
# # """

# # # ----- Load Model -----
# # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# # model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)

# # # ----- Generate Answer -----
# # inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
# # with torch.no_grad():
# #     outputs = model.generate(**inputs, max_new_tokens=150)

# # answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

# # # ----- Output -----
# # print("--- Final Answer \n")
# # print(answer)

# #-------------------------------------------------------------------- Mistral---------------------------------------------------------------------------------------------------------------------

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import pandas as pd

# # --------- CONFIG ---------
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
# RERANKED_FILE = "reranked_results.csv"
# TOP_K = 3
# # QUERY = "Can I return customized or engraved items?" ---------------------------------------gOOD ONE
# QUERY = "Can I return items bought during holiday sales?"

# # --------- LOAD MODEL ---------
# print("Loading Mistral-7B-Instruct...")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
# model.eval()

# print("Hello")
# # --------- LOAD CHUNKS ---------
# df = pd.read_csv(RERANKED_FILE)
# top_chunks = df.sort_values(by="rerank_score", ascending=False).head(TOP_K)["chunk_text"].tolist()

# # --------- BUILD PROMPT ---------
# context = "\n".join([f"- {chunk}" for chunk in top_chunks])
# prompt = f"""[INST] You are a helpful assistant answering customer queries using the company's return policy.
# Context:
# {context}

# Question: {QUERY}
# Answer concisely and helpfully. [/INST]
# """

# # --------- GENERATE ANSWER ---------
# inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

# with torch.no_grad():
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=150,
#         temperature=0.7,
#         top_p=0.9,
#         do_sample=True,
#         repetition_penalty=1.1,
#         eos_token_id=tokenizer.eos_token_id
#     )

# answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print("\n--- Final Answer ---\n")
# # print(answer.split("Answer:")[-1].strip())
# final = answer
# if "Answer:" in final:
#     final = final.split("Answer:")[-1]
# if "[/INST]" in final:
#     final = final.split("[/INST]")[-1]
# print(final.strip())

