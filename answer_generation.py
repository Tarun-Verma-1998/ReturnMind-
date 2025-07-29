#----------------------------------------------------------------Flan-t5-----------------------------------------------------------

# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import torch

# # ----- Config -----
# QUERY = "Can I return customized or engraved items?"
# TOP_K = 3
# MODEL_NAME = "google/flan-t5-base"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # ----- Load Reranked Chunks -----
# df = pd.read_csv("reranked_results.csv")
# top_chunks = df.sort_values("rerank_score", ascending=False).head(TOP_K)["chunk_text"].tolist()

# # ----- Multi-shot Format -----
# passages = "\n\n".join([f"Passage {i+1}: {chunk}" for i, chunk in enumerate(top_chunks)])
# prompt = f"""
# You are a helpful assistant at ReturnMind.

# Below are some passages from ReturnMind's return policy.

# Your task is to:
# 1. Find the passage most relevant to the customer's question.
# 2. Answer the question using only that passage.
# 3. If the information is not present, say:
#    "Sorry, I couldn't find this information in the policy."

# {passages}

# Customer Query: {QUERY}
# Answer:
# """

# # ----- Load Model -----
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)

# # ----- Generate Answer -----
# inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
# with torch.no_grad():
#     outputs = model.generate(**inputs, max_new_tokens=150)

# answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

# # ----- Output -----
# print("--- Final Answer \n")
# print(answer)


#-------------------------------------------------------------------- Mistral---------------------------------------------------------------------------------------------------------------------

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

# --------- CONFIG ---------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
RERANKED_FILE = "reranked_results.csv"
TOP_K = 3
# QUERY = "Can I return customized or engraved items?" ---------------------------------------gOOD ONE
QUERY = "Can I return items bought during holiday sales?"

# --------- LOAD MODEL ---------
print("Loading Mistral-7B-Instruct...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
model.eval()

print("Hello")
# --------- LOAD CHUNKS ---------
df = pd.read_csv(RERANKED_FILE)
top_chunks = df.sort_values(by="rerank_score", ascending=False).head(TOP_K)["chunk_text"].tolist()

# --------- BUILD PROMPT ---------
context = "\n".join([f"- {chunk}" for chunk in top_chunks])
prompt = f"""[INST] You are a helpful assistant answering customer queries using the company's return policy.
Context:
{context}

Question: {QUERY}
Answer concisely and helpfully. [/INST]
"""

# --------- GENERATE ANSWER ---------
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
print("\n--- Final Answer ---\n")
# print(answer.split("Answer:")[-1].strip())
final = answer
if "Answer:" in final:
    final = final.split("Answer:")[-1]
if "[/INST]" in final:
    final = final.split("[/INST]")[-1]
print(final.strip())

