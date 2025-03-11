import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import normalize
import nltk
from nltk.tokenize import word_tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from rich import print
from rich.panel import Panel
from rich.table import Table

nltk.download('punkt')

# Load financial data from JSON file
file_path = "financial_data.json"
with open(file_path, "r") as file:
    data = json.load(file)

# Extract and format financial data into text chunks
def extract_text_chunks(data, chunk_size=2):
    chunks = []
    for section, values in data.items():
        if isinstance(values, dict):
            for key, sub_values in values.items():
                if isinstance(sub_values, dict):
                    for date, value in sub_values.items():
                        chunks.append(f"{key} ({date}): {value}")
                else:
                    chunks.append(f"{key}: {sub_values}")
        else:
            chunks.append(f"{section}: {values}")
    return chunks

text_chunks = extract_text_chunks(data)

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Generate embeddings and create FAISS index
embeddings = embedding_model.encode(text_chunks, convert_to_numpy=True)
embeddings = normalize(embeddings, axis=1, norm='l2')
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# BM25 keyword-based retrieval
tokenized_corpus = [word_tokenize(doc.lower()) for doc in text_chunks]
bm25 = BM25Okapi(tokenized_corpus)

# Load small open-source LLM
lm_model_name = "facebook/opt-125m"
lm_tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
lm_model = AutoModelForCausalLM.from_pretrained(
    lm_model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True
)

def retrieve_financial_info(query, top_k=2):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    query_embedding = normalize(query_embedding, axis=1, norm='l2')

    # FAISS search
    _, faiss_indices = index.search(query_embedding, top_k)
    faiss_results = [text_chunks[idx] for idx in faiss_indices[0]]

    # BM25 search
    tokenized_query = word_tokenize(query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k]
    bm25_results = [text_chunks[idx] for idx in bm25_top_indices]

    # Combine results, remove duplicates, and return top matches
    combined_results = list(set(faiss_results + bm25_results))
    return combined_results[:top_k]

def generate_response(query, retrieved_docs):
    context = "\n".join(retrieved_docs)
    prompt = f"Answer based on the financial data:\n{context}\n\nQuery: {query}\nAnswer:"
    inputs = lm_tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = lm_model.generate(**inputs, max_new_tokens=50)

    return lm_tokenizer.decode(output[0], skip_special_tokens=True)

# Example query
query = "What was Apple's net income in 2023?"
retrieved_info = retrieve_financial_info(query)
response = generate_response(query, retrieved_info)

# Format retrieved financial data table
table = Table(title="🔍 Apple (AAPL) Financial Data", show_lines=True, pad_edge=False)
table.add_column("Index", style="bold cyan", justify="center", width=5)
table.add_column("Metric", style="bold yellow", width=40)
table.add_column("Value", style="bold green", width=20)

for i, info in enumerate(retrieved_info, 1):
    if ": " in info:
        metric, value = info.split(": ", 1)
        formatted_value = f"${float(value):,.2f}".replace(".00", "")  # Convert to readable format
    else:
        metric, formatted_value = info, "N/A"
    table.add_row(str(i), metric.strip(), formatted_value)

# Display retrieved information and AI response
print("\n")
print(Panel.fit(table, title="[bold green]📊 Financial Data Retrieval[/bold green]", border_style="green"))

print("\n")
print(Panel.fit(f"[bold magenta]🤖 AI-Generated Response:[/bold magenta]\n\n[italic white]{response}",
            title="[bold blue]📢 AI Answer[/bold blue]", border_style="bright_yellow"))
print("\n")
