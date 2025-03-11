import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load financial data from JSON file
with open("financial_data.json", "r") as file:
    data = json.load(file)

# Extract and format financial data into text chunks
def extract_text_chunks(data):
    chunks = []
    for section, values in data.items():
        if isinstance(values, dict):
            for key, sub_values in values.items():
                if isinstance(sub_values, dict):
                    for date, value in sub_values.items():
                        chunks.append(f"{section} - {key} ({date}): {value}")
                else:
                    chunks.append(f"{section} - {key}: {sub_values}")
        else:
            chunks.append(f"{section}: {values}")
    return chunks

text_chunks = extract_text_chunks(data)

# Load pre-trained sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
embeddings = model.encode(text_chunks, convert_to_numpy=True)

# Create FAISS index
d = embeddings.shape[1]  # Dimension of embeddings
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# Define retrieval function
def retrieve_financial_info(query, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    results = [text_chunks[idx] for idx in indices[0]]
    return results

# Example query
query = "What was Apple's net income in 2023?"
retrieved_info = retrieve_financial_info(query)

# Print retrieved financial information
for i, info in enumerate(retrieved_info, 1):
    print(f"{i}. {info}")
