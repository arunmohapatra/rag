import json
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import normalize
import nltk
from nltk.tokenize import word_tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

nltk.download('punkt')

# Load financial data
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
    faiss_distances, faiss_indices = index.search(query_embedding, top_k)
    faiss_results = [(text_chunks[idx], 1 - (dist / np.max(faiss_distances))) for idx, dist in zip(faiss_indices[0], faiss_distances[0])]

    # BM25 search
    tokenized_query = word_tokenize(query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k]
    bm25_results = [(text_chunks[idx], bm25_scores[idx] / np.max(bm25_scores)) for idx in bm25_top_indices]

    # Merge results, average confidence scores
    result_dict = {}
    for text, score in faiss_results + bm25_results:
        if text in result_dict:
            result_dict[text] = (result_dict[text] + score) / 2  # Average score if exists
        else:
            result_dict[text] = score

    # Sort by confidence score
    sorted_results = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]  # Return top results with confidence

def generate_response(query, retrieved_docs):
    formatted_docs = []

    for info, confidence in retrieved_docs:
        if ": " in info:
            metric, value = info.split(": ", 1)
            value = float(value)

            # Format value with colors
            formatted_value = f"${value:,.2f}".replace(".00", "")
            color = "🟢" if value > 0 else "🔴" if value < 0 else "🔵"

            formatted_docs.append(f"{color} **{metric.strip()}**: `{formatted_value}` (Confidence: `{confidence:.2%}`)")
        else:
            formatted_docs.append(f"🔵 **{info.strip()}** (Confidence: `{confidence:.2%}`)")

    # Combine formatted financial data
    formatted_context = "<br>".join(formatted_docs)

    # Modify prompt for AI
    prompt = f"Based on the following financial data, answer the query:\n\n{formatted_context}\n\nQuery: {query}\nAnswer:"

    inputs = lm_tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = lm_model.generate(**inputs, max_new_tokens=50)

    ai_response = lm_tokenizer.decode(output[0], skip_special_tokens=True)

    # Format AI response for Streamlit
    final_response = f"""
    <div style="background-color:#f4f4f4; padding:15px; border-radius:10px; font-size:16px;">
    <b>📊 Relevant Financial Data:</b><br>{formatted_context}
    <hr>
    <b>🤖 AI-Generated Response:</b><br>
    <i>{ai_response}</i>
    </div>
    """

    return final_response

# Streamlit UI
st.set_page_config(page_title="Financial RAG Model", layout="wide")
st.title("📊 Financial RAG Model - AI-Driven Insights")

# Query Input
query = st.text_input("🔍 Enter your financial query:", "What was Apple's net income in 2023?")
if st.button("Get Answer"):
    with st.spinner("Retrieving relevant financial information..."):
        retrieved_info = retrieve_financial_info(query)
        response = generate_response(query, retrieved_info)

        # Display Retrieved Information & AI Response with Colors
        st.subheader("📌 Retrieved Financial Information")
        st.markdown(response, unsafe_allow_html=True)
