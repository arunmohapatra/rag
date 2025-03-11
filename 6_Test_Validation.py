import re
import json
import os
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

# ✅ Fix: Ensure NLTK works correctly on Streamlit Cloud
NLTK_PATH = os.path.expanduser("~/.nltk_data")
os.makedirs(NLTK_PATH, exist_ok=True)
os.environ["NLTK_DATA"] = NLTK_PATH
nltk.data.path.append(NLTK_PATH)

# ✅ Ensure 'punkt' is downloaded to avoid errors
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=NLTK_PATH)

# ✅ Ensure financial data file exists
file_path = "financial_data.json"
if not os.path.exists(file_path):
    st.error("Error: Missing `financial_data.json` file. Please upload it to your repository.")
    st.stop()

with open(file_path, "r") as file:
    data = json.load(file)

def extract_text_chunks(data):
    """Extract financial data into text chunks for retrieval."""
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

# ✅ Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedding_model.encode(text_chunks, convert_to_numpy=True)
embeddings = normalize(embeddings, axis=1, norm='l2')

# ✅ Initialize FAISS for similarity search
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# ✅ BM25 Tokenization for better search
tokenized_corpus = [word_tokenize(doc.lower()) for doc in text_chunks]
bm25 = BM25Okapi(tokenized_corpus)

# ✅ Load Transformer Model (Fixed `low_cpu_mem_usage` issue)
lm_model_name = "facebook/opt-125m"
lm_tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
lm_model = AutoModelForCausalLM.from_pretrained(
    lm_model_name, torch_dtype=torch.float32  # Removed `low_cpu_mem_usage=True`
)

# ✅ Input-side Guardrail: Validate user queries
def validate_query(query):
    financial_keywords = ["revenue", "profit", "loss", "net income", "earnings", "expense", "cash flow"]
    prohibited_patterns = [r"(personal address|credit card|SSN|passport)", r"\b(?:capital of|weather in)\b"]

    if any(re.search(pattern, query, re.IGNORECASE) for pattern in prohibited_patterns):
        return False, "❌ This query is not allowed. Please ask financial-related questions."

    if not any(keyword in query.lower() for keyword in financial_keywords):
        return False, "⚠️ This query does not appear financial-related. Try asking about earnings, revenue, or expenses."

    return True, "✅ Query accepted. Processing..."

# ✅ Financial Retrieval: FAISS + BM25
def retrieve_financial_info(query, top_k=2):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    query_embedding = normalize(query_embedding, axis=1, norm='l2')

    # FAISS retrieval
    faiss_distances, faiss_indices = index.search(query_embedding, top_k)
    faiss_results = [(text_chunks[idx], 1 - (dist / np.max(faiss_distances))) for idx, dist in zip(faiss_indices[0], faiss_distances[0])]

    # BM25 retrieval
    tokenized_query = word_tokenize(query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k]
    bm25_results = [(text_chunks[idx], bm25_scores[idx] / np.max(bm25_scores)) for idx in bm25_top_indices]

    # Combine FAISS & BM25 results
    result_dict = {}
    for text, score in faiss_results + bm25_results:
        result_dict[text] = (result_dict[text] + score) / 2 if text in result_dict else score

    sorted_results = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]

# ✅ Output-side Guardrail: Filter hallucinations
def filter_response(ai_response, retrieved_docs):
    relevant_data = [doc[0] for doc in retrieved_docs]
    if any(keyword in ai_response.lower() for keyword in relevant_data):
        return ai_response
    return "⚠️ AI-generated response lacks supporting financial data. Unable to provide a confident answer."

# ✅ AI Response Generation
def generate_response(query, retrieved_docs):
    formatted_context = "<br>".join([f"**{doc[0]}** (Confidence: `{doc[1]:.2%}`)" for doc in retrieved_docs])
    prompt = f"Based on the following financial data, answer the query:\n\n{formatted_context}\n\nQuery: {query}\nAnswer:"
    inputs = lm_tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = lm_model.generate(**inputs, max_new_tokens=50)

    ai_response = lm_tokenizer.decode(output[0], skip_special_tokens=True)
    filtered_response = filter_response(ai_response, retrieved_docs)

    return f"<b>📊 Retrieved Financial Data:</b><br>{formatted_context}<hr><b>🤖 AI-Generated Response:</b><br><i>{filtered_response}</i>"

# ✅ Streamlit UI
st.set_page_config(page_title="Financial RAG Model", layout="wide")
st.title("📊 Financial RAG Model - AI-Driven Insights")

query = st.text_input("🔍 Enter your financial query:", "What was Apple's net income in 2023?")
if st.button("Get Answer"):
    is_valid, validation_message = validate_query(query)
    if not is_valid:
        st.error(validation_message)
    else:
        with st.spinner("Retrieving relevant financial information..."):
            retrieved_info = retrieve_financial_info(query)
            response = generate_response(query, retrieved_info)
            st.markdown(response, unsafe_allow_html=True)
