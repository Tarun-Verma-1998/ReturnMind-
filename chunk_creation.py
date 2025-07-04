import os
import re
import spacy
import pandas as pd

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Folder containing your .txt files
BASE_DOCS_PATH = "BaseDocs"

# Function to clean text (minimal cleaning)
def clean_text(text):
    text = text.replace('\xa0', ' ')  # Replace non-breaking space
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)  # Remove Markdown headers like ### Title
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold markdown (**bold** → bold)
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters (™ ε etc.)
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces/tabs/newlines
    return text.strip()

# Sentence splitter using spaCy
def get_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

# Chunking function using sliding window over sentences
def chunk_document(doc_name, text, word_limit=400, overlap=2):
    sentences = get_sentences(clean_text(text))
    chunks = []
    chunk_id = 0
    i = 0
    while i < len(sentences):
        chunk_sents = []
        word_count = 0
        j = i
        while j < len(sentences) and word_count < word_limit:
            chunk_sents.append(sentences[j])
            word_count += len(sentences[j].split())
            j += 1
        chunk_text = " ".join(chunk_sents)
        chunks.append((doc_name, chunk_id, chunk_text))
        chunk_id += 1
        i += (j - i - overlap) if (j - i - overlap) > 0 else 1
    return chunks

# Process all documents in BaseDocs/
all_chunks = []
for file_name in os.listdir(BASE_DOCS_PATH):
    if file_name.endswith(".txt"):
        file_path = os.path.join(BASE_DOCS_PATH, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_document(file_name, text)
        all_chunks.extend(chunks)

# Save to CSV
df = pd.DataFrame(all_chunks, columns=["doc_name", "chunk_id", "chunk_text"])
df.to_csv("chunked_documents.csv", index=False)

print("Documents are chunked successfully..")
