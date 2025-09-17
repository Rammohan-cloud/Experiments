import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from gpt4all import GPT4All
from sympy import false
import unicodedata

# ---------------------------
# Load models once
# ---------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("./models/paraphrase-MiniLM-L3-v2")

@st.cache_resource
def load_gpt4all():
    return GPT4All("gpt4all-falcon.Q4_0.gguf", model_path="./models", allow_download=false)

# ---------------------------
# Load CSV
# ---------------------------
@st.cache_data
def load_csv(file):
    df = pd.read_csv(file)
    df = df.fillna('')
    return df

# ---------------------------
# Normalize cell / row text
# ---------------------------
def normalize_cell(x):
    x_str = str(x)
    x_str = unicodedata.normalize("NFKD", x_str)
    x_str = x_str.replace("\xa0", " ")
    x_str = " ".join(x_str.split())
    return x_str.lower()

def row_to_text(row):
    return " ".join([str(cell) for cell in row.values])

# ---------------------------
# Build FAISS index
# ---------------------------
@st.cache_resource
def build_faiss_index(df):
    embedder = load_embedder()
    chunks = [row_to_text(row) for idx, row in df.iterrows()]
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))
    return index, embeddings, chunks

# ---------------------------
# Semantic search
# ---------------------------
def semantic_search(query, df, index, chunks, top_k=20):
    embedder = load_embedder()
    q_vec = embedder.encode([query])
    q_vec = q_vec / np.linalg.norm(q_vec, axis=1, keepdims=True)
    _, idx = index.search(np.array(q_vec).astype("float32"), top_k)
    matched_rows = df.iloc[idx[0]]
    return matched_rows

# ---------------------------
# Exact match search (any column)
# ---------------------------
def exact_match_search(query, df):
    query_norm = normalize_cell(query)
    mask = df.apply(lambda row: row.astype(str).apply(normalize_cell).str.contains(query_norm).any(), axis=1)
    return df[mask]

# ---------------------------
# Merge semantic + exact matches
# ---------------------------
def combined_search(query, df, index, chunks, top_k=20):
    sem_rows = semantic_search(query, df, index, chunks, top_k)
    exact_rows = exact_match_search(query, df)
    combined = pd.concat([sem_rows, exact_rows]).drop_duplicates()
    return combined

# ---------------------------
# Row-wise generative prompt
# ---------------------------
def generate_gpt_answer(query, matched_rows):
    context = "\n".join([", ".join(f"{col}: {val}" for col, val in zip(matched_rows.columns, row))
                         for row in matched_rows.values])
    prompt = f"""
You are a helpful assistant.
For each row in the following catalog data, describe the product and all its details in a complete sentence.
Include numeric info (Price, Quantity, etc.) exactly as provided.
Do NOT summarize multiple rows; describe each row separately.
If no relevant data is present, reply exactly: "I don't know".

Catalog Data:
{context}

Question: {query}
Answer:
"""
    gpt = load_gpt4all()
    with gpt.chat_session():
        ans = gpt.generate(prompt, max_tokens=800)
    return ans.strip()

# ---------------------------
# Streamlit App
# ---------------------------
st.title("📊 CSV Q&A with Full Row-wise Generative Answers")

uploaded = st.file_uploader("Upload CSV", type="csv")
if uploaded:
    df = load_csv(uploaded)
    st.success(f"✅ CSV loaded successfully with {len(df)} rows!")

    # Build FAISS index
    index, embeddings, chunks = build_faiss_index(df)

    query = st.text_input("Ask a question about the catalog:")
    if query:
        matched_rows = combined_search(query, df, index, chunks, top_k=20)
        if matched_rows.empty:
            st.warning("No relevant data found in CSV.")
        else:
            # Step 1: Optionally, compute numeric info
            num_ans = None
            if 'Price' in matched_rows.columns:
                q = query.lower()
                if 'average price' in q:
                    avg = pd.to_numeric(matched_rows['Price'], errors='coerce').mean()
                    num_ans = f"The average price is ${avg:.2f}."
                elif 'minimum price' in q or 'lowest price' in q:
                    min_val = pd.to_numeric(matched_rows['Price'], errors='coerce').min()
                    num_ans = f"The lowest price is ${min_val:.2f}."
                elif 'maximum price' in q or 'highest price' in q:
                    max_val = pd.to_numeric(matched_rows['Price'], errors='coerce').max()
                    num_ans = f"The highest price is ${max_val:.2f}."
                elif 'how many' in q or 'number of' in q:
                    num_ans = f"There are {len(matched_rows)} matching products."

            if num_ans:
                st.markdown("### 💡 Computed Answer")
                st.write(num_ans)

            # Step 2: Row-wise generative answer
            gen_ans = generate_gpt_answer(query, matched_rows)
            st.markdown("### 💡 Generative Answer (Row-wise)")
            st.write(gen_ans)
