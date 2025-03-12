import os
import json
import yfinance as yf
import pandas as pd
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
from collections import deque

# ✅ Must be the first Streamlit command
st.set_page_config(page_title="Financial RAG Model", layout="wide")

# ✅ Ensure NLTK resources are available
nltk.download('punkt')

# ===========================
# 🛡 Guardrails: Input Validation
# ===========================

def is_valid_query(query):
    """Validate user query to prevent irrelevant or harmful inputs."""
    finance_keywords = ["revenue", "profit", "net income", "expenses", "earnings", "stock", "market"]
    blacklist = ["hack", "attack", "password", "bomb", "violence", "kill", "harm", "illegal"]

    query_lower = query.lower()

    # Block offensive or harmful content
    if any(word in query_lower for word in blacklist):
        return False, "🚨 Your query contains restricted terms."

    # Ensure query is relevant to finance
    if not any(word in query_lower for word in finance_keywords):
        return False, "⚠️ Please ask a finance-related question."

    return True, ""

# ===========================
# 🔹 Step 1: Fetch & Structure Financial Data
# ===========================

@st.cache_data
def fetch_financial_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        income_stmt = stock.financials.T
        balance_sheet = stock.balance_sheet.T
        cash_flow = stock.cashflow.T

        # ✅ Convert index (timestamps) to string
        income_stmt.index = income_stmt.index.astype(str)
        balance_sheet.index = balance_sheet.index.astype(str)
        cash_flow.index = cash_flow.index.astype(str)

        structured_data = {
            "ticker": ticker,
            "income_statement": income_stmt.to_dict(),
            "balance_sheet": balance_sheet.to_dict(),
            "cash_flow": cash_flow.to_dict()
        }

        # ✅ Ensure JSON is saved in the working directory
        json_file_path = os.path.join(os.getcwd(), "financial_data.json")
        with open(json_file_path, "w") as f:
            json.dump(structured_data, f, indent=4, default=str)

        return structured_data

    except Exception as e:
        st.error(f"⚠️ Error fetching data for {ticker}: {e}")
        return None

# ✅ Ensure financial data is available
# file_path = "financial_data.json"
file_path = os.path.join(os.getcwd(), "financial_data.json")

# ✅ Auto-delete corrupted JSON & regenerate
if os.path.exists(file_path):
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
    except (json.JSONDecodeError, FileNotFoundError):
        os.remove(file_path)
        st.warning("⚠️ Invalid JSON detected. Fetching fresh financial data for AAPL...")
        data = fetch_financial_data("AAPL")
else:
    st.warning("📌 No financial data found. Fetching default data for AAPL...")
    data = fetch_financial_data("AAPL")

if data is None:
    st.error("⚠️ Financial data could not be fetched. Please try another stock ticker.")
    st.stop()

# ===========================
# 🔹 Step 2: Memory-Augmented Retrieval
# ===========================

@st.cache_resource
def setup_rag_system(data):
    text_chunks = []
    for section, values in data.items():
        if isinstance(values, dict):
            for key, sub_values in values.items():
                if isinstance(sub_values, dict):
                    for date, value in sub_values.items():
                        text_chunks.append(f"{key} ({date}): {value}")
                else:
                    text_chunks.append(f"{key}: {sub_values}")
        else:
            text_chunks.append(f"{section}: {values}")

    # ✅ Load embedding model
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # ✅ Generate embeddings and create FAISS index
    embeddings = embedding_model.encode(text_chunks, convert_to_numpy=True)
    embeddings = normalize(embeddings, axis=1, norm='l2')
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    # ✅ BM25 keyword-based retrieval
    tokenized_corpus = [word_tokenize(doc.lower()) for doc in text_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    return embedding_model, index, bm25, text_chunks

embedding_model, index, bm25, text_chunks = setup_rag_system(data)

# ✅ Memory Storage for Multi-Turn Queries
MEMORY_SIZE = 5
memory_store = deque(maxlen=MEMORY_SIZE)

def retrieve_financial_info(query, top_k=2):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    query_embedding = normalize(query_embedding, axis=1, norm='l2')

    # 🔹 FAISS search
    faiss_distances, faiss_indices = index.search(query_embedding, top_k)
    faiss_results = [(text_chunks[idx], 1 - (dist / np.max(faiss_distances))) for idx, dist in zip(faiss_indices[0], faiss_distances[0])]

    # 🔹 BM25 search
    # tokenized_query = word_tokenize(query.lower())
    # bm25_scores = bm25.get_scores(tokenized_query)
    # bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k]
    # bm25_results = [(text_chunks[idx], bm25_scores[idx] / np.max(bm25_scores)) for idx in bm25_top_indices]
    tokenized_query = word_tokenize(query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)

    # ✅ Avoid division by zero
    bm25_max_score = np.max(bm25_scores)
    if bm25_max_score == 0 or np.isnan(bm25_max_score):  
        bm25_results = [(text_chunks[idx], 0) for idx in np.argsort(bm25_scores)[::-1][:top_k]]  # Assign zero confidence
    else:
        bm25_results = [(text_chunks[idx], bm25_scores[idx] / bm25_max_score) for idx in np.argsort(bm25_scores)[::-1][:top_k]]


    # 🔹 Memory-Augmented Retrieval
    memory_results = []
    for prev_query, prev_docs in memory_store:
        if query.lower() in prev_query.lower():
            memory_results.extend(prev_docs)

    # 🔹 Combine Results
    result_dict = {}
    for text, score in faiss_results + bm25_results + memory_results:
        result_dict[text] = max(result_dict.get(text, 0), score)

    sorted_results = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
    final_results = sorted_results[:top_k]

    # 🔹 Store in Memory
    memory_store.append((query, final_results))

    return final_results

# ===========================
# 🔹 Step 3: AI Response Generation
# ===========================

@st.cache_resource
def load_model():
    lm_model_name = "facebook/opt-125m"
    lm_tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
    lm_model = AutoModelForCausalLM.from_pretrained(
        lm_model_name, torch_dtype=torch.float32
    )
    return lm_model, lm_tokenizer

lm_model, lm_tokenizer = load_model()

def generate_response(query, retrieved_docs):
    memory_context = "\n".join([doc[0] for _, docs in memory_store for doc in docs])
    context = "\n".join([doc[0] for doc in retrieved_docs])

    # 🔹 Generate prompt with memory
    prompt = f"Previous relevant data:\n{memory_context}\n\nCurrent data:\n{context}\n\nQuery: {query}\nAnswer:"
    inputs = lm_tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = lm_model.generate(**inputs, max_new_tokens=50)

    return lm_tokenizer.decode(output[0], skip_special_tokens=True)

# ===========================
# 🔹 Step 4: Streamlit UI
# ===========================

st.title("📊 Financial RAG Model - AI-Driven Insights")

# ✅ User selects stock ticker
ticker = st.sidebar.text_input("📈 Enter Stock Ticker (e.g., AAPL, TSLA, MSFT):", "AAPL")

if st.sidebar.button("Fetch Data"):
    st.session_state["financial_data"] = fetch_financial_data(ticker)
    st.success(f"✅ Data fetched for {ticker}")

# ✅ Ensure data is loaded
if "financial_data" not in st.session_state:
    st.session_state["financial_data"] = fetch_financial_data("AAPL")

data = st.session_state["financial_data"]

# ✅ User Query Input
query = st.text_input("🔍 Enter your financial query:", "What was Apple's net income in 2023?")

if st.button("Get Answer"):
    with st.spinner("Retrieving relevant financial information..."):
        is_valid, warning_message = is_valid_query(query)
        if not is_valid:
            st.warning(warning_message)
        else:
            retrieved_info = retrieve_financial_info(query)
            response = generate_response(query, retrieved_info)

            # ✅ Display Retrieved Financial Data
            st.subheader("📊 Retrieved Financial Data:")
            if not retrieved_info:
                st.warning("⚠ No relevant financial data retrieved. Try a different query.")
            else:
                for i, (doc, score) in enumerate(retrieved_info, 1):
                    st.markdown(f"**{i}. {doc}**  \n*Confidence Score:* `{score:.2%}`")

            # ✅ Display AI Response
            st.subheader("🤖 AI-Generated Answer:")
            st.markdown(f"> {response}")