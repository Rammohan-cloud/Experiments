import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from gpt4all import GPT4All
import re
import faiss
import time

# -------------------------------
# Load Models
# -------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("./models/paraphrase-MiniLM-L3-v2")

@st.cache_resource
def load_gpt4all():
    return GPT4All("gpt4all-falcon.Q4_0.gguf", model_path="./models", allow_download=False)

# -------------------------------
# Embedding Utilities
# -------------------------------
def embed_text_in_batches(text_list, embedder, batch_size=32):
    embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        batch_emb = embedder.encode(batch, convert_to_numpy=True)
        embeddings.append(batch_emb)
    return np.vstack(embeddings)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

# -------------------------------
# PDF Handling
# -------------------------------
def extract_pdf_chunks(pdf_file, chunk_size=500, overlap=50):
    pdf = PdfReader(pdf_file)
    text = "".join([page.extract_text() or "" for page in pdf.pages])
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

@st.cache_data
def get_pdf_embeddings(pdf_file, _embedder):
    chunks = extract_pdf_chunks(pdf_file)
    chunk_embeddings = embed_text_in_batches(chunks, _embedder, batch_size=32)
    return chunks, chunk_embeddings

def get_top_sentence(chunk_text, question, embedder, max_sentences=5):
    sentences = re.split(r'(?<=[.!?]) +', chunk_text)
    sentences = sentences[:max_sentences]
    sent_embeddings = embed_text_in_batches(sentences, embedder, batch_size=16)
    q_emb = embed_text_in_batches([question], embedder)
    sent_sims = cos_sim(q_emb, sent_embeddings)[0].cpu().numpy()
    max_idx = np.argmax(sent_sims)
    return sentences[max_idx]

# -------------------------------
# CSV Handling
# -------------------------------
@st.cache_data
def get_csv_embeddings(df, _embedder):
    row_texts = df.apply(lambda row: " | ".join(map(str, row.values)), axis=1).tolist()
    embeddings = embed_text_in_batches(row_texts, _embedder, batch_size=32)
    return row_texts, embeddings

# -------------------------------
# Streamlit UI Enhancements
# -------------------------------
st.set_page_config(page_title="PDF & CSV QnA Optimized", layout="wide")

# Custom CSS for professional styling
st.markdown("""
<style>
    body {
        font-family: 'Segoe UI', sans-serif;
    }
    .main {
        background-color: #f9fafb;
        padding: 1rem;
    }
    h1, h2, h3 {
        font-weight: 600;
        color: #1f2937;
    }
    .stTabs [role="tablist"] button {
        background-color: #e5e7eb;
        border-radius: 10px;
        margin-right: 10px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        color: #111827;
    }
    .stTabs [role="tablist"] button[aria-selected="true"] {
        background-color: #2563eb;
        color: white !important;
    }
    .result-box {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1rem;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
    }
    .status-text {
        font-size: 0.9rem;
        color: #6b7280;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>📘 PDF & 📊 CSV Q&A Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6b7280;'>Upload documents and ask smart questions with AI-powered search.</p>", unsafe_allow_html=True)

embedder = load_embedder()
model = load_gpt4all()

tab_pdf, tab_csv = st.tabs(["📘 PDF Q&A", "📊 CSV Q&A"])

MAX_CONTEXT_CHARS = 3000  # Prevent exceeding model context window

# -------------------------------
# PDF Tab
# -------------------------------
with tab_pdf:
    st.markdown("### 📄 Upload PDF Document")
    pdf_file = st.file_uploader("Choose a PDF", type="pdf")

    if pdf_file:
        start_time = time.time()
        chunks, chunk_embeddings = get_pdf_embeddings(pdf_file, embedder)
        faiss_index = build_faiss_index(chunk_embeddings)
        st.success(f"✅ PDF processed into {len(chunks)} chunks.")

        question = st.text_input("🔍 Ask a question about the PDF:")

        if question:
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Encoding question...")
            q_emb = embed_text_in_batches([question], embedder)
            faiss.normalize_L2(q_emb)
            progress_bar.progress(20)

            status_text.text("Retrieving top chunks via FAISS...")
            k = min(30, len(chunks))
            D, I = faiss_index.search(q_emb, k)
            faiss_chunks = [chunks[i] for i, sim in zip(I[0], D[0]) if sim > 0.1]
            progress_bar.progress(40)

            keyword_chunks = [c for c in chunks if re.search(re.escape(question), c, re.IGNORECASE)]
            relevant_chunks = list(dict.fromkeys(faiss_chunks + keyword_chunks))[:5]

            if not relevant_chunks:
                progress_bar.progress(100)
                status_text.text("No relevant chunks found.")
                st.markdown("<div class='result-box'><b>📖 Answer:</b><br>The answer is not present in the PDF.</div>", unsafe_allow_html=True)
            else:
                status_text.text("Highlighting top sentences...")
                for rank, chunk_text in enumerate(relevant_chunks, start=1):
                    highlighted_sentence = get_top_sentence(chunk_text, question, embedder)
                    display_text = chunk_text.replace(highlighted_sentence, f"**{highlighted_sentence}**")
                    preview_words = 30
                    chunk_preview = " ".join(display_text.split()[:preview_words])
                    st.markdown(f"<div class='result-box'><b>Chunk {rank}:</b><br>{chunk_preview} ...</div>", unsafe_allow_html=True)
                progress_bar.progress(70)

                context = " ".join(relevant_chunks)[:MAX_CONTEXT_CHARS]

                status_text.text("Generating answer...")
                prompt = f"""
You are a strict assistant. Answer the question based ONLY on the PDF content below.
- If answer exists, give it concisely.
- If partial, summarize clearly.
- If not present, say: "The answer is not present in the PDF."

Context:
{context}

Question: {question}
Answer:
"""
                response = model.generate(prompt, max_tokens=200)
                progress_bar.progress(100)
                status_text.text("Done!")

                end_time = time.time()
                st.markdown(f"<div class='result-box'><b>📖 Answer:</b><br>{response}</div>", unsafe_allow_html=True)
                st.caption(f"⏱ Total time taken: {end_time - start_time:.2f} seconds")

# -------------------------------
# CSV Tab
# -------------------------------
def clean_features(features, max_items=10):
    words = re.split(r'[,;]', str(features))
    seen, cleaned = set(), []
    for w in words:
        w_norm = w.strip().lower()
        if w_norm and w_norm not in seen:
            cleaned.append(w.strip().capitalize())
            seen.add(w_norm)
        if len(cleaned) >= max_items:
            break
    return ", ".join(cleaned)

with tab_csv:
    st.markdown("### 📊 Upload CSV Data")
    csv_file = st.file_uploader("Choose a CSV", type="csv")

    if csv_file:
        start_time = time.time()
        df = pd.read_csv(csv_file)
        st.success(f"✅ CSV loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
        st.markdown("<div class='result-box'>📋 <b>Preview of Data</b></div>", unsafe_allow_html=True)
        st.dataframe(df.head(10))

        question = st.text_input("🔍 Ask a question about the CSV:")

        if question:
            progress_bar = st.progress(0)
            status_text = st.empty()
            progress_bar.progress(10)

            status_text.text("Filtering relevant rows...")
            mask = df.astype(str).apply(lambda col: col.str.contains(question, case=False, na=False))
            relevant_df = df[mask.any(axis=1)]
            progress_bar.progress(30)

            if relevant_df.empty:
                progress_bar.progress(100)
                st.markdown("<div class='result-box'><b>📊 Answer:</b><br>No relevant data found.</div>", unsafe_allow_html=True)
            else:
                if "ProductId" in relevant_df.columns and "ProductName" in relevant_df.columns:
                    relevant_df = relevant_df.drop_duplicates(subset=["ProductId", "ProductName"])
                else:
                    relevant_df = relevant_df.drop_duplicates()

                relevant_df = relevant_df.head(5)
                progress_bar.progress(50)

                structured_rows = []
                for _, row in relevant_df.iterrows():
                    structured_rows.append({
                        "ProductName": row.get("ProductName", ""),
                        "Price": row.get("Price", ""),
                        "Features": clean_features(row.get("Features", ""), max_items=10),
                        "CustomizationOptions": clean_features(row.get("CustomizationOptions", ""), max_items=10),
                        "MaterialOrFormat": clean_features(row.get("MaterialOrFormat", ""), max_items=5),
                        "UseCase": clean_features(row.get("UseCase", ""), max_items=5)
                    })
                progress_bar.progress(70)

                summary_lines = [
                    f"{r['ProductName']} | Price: {r['Price']} | Features: {r['Features']} | "
                    f"Customization: {r['CustomizationOptions']} | Material/Format: {r['MaterialOrFormat']} | "
                    f"Use Case: {r['UseCase']}"
                    for r in structured_rows
                ]
                summary_text = "\n".join(summary_lines)
                progress_bar.progress(80)

                status_text.text("Generating answer...")
                prompt = f"""
You are a helpful assistant. Present the product details clearly without any duplication.
Ensure Features, Customization, Material, and Use Case are concise and customer-friendly.
Keep product names and prices intact.

{summary_text}

Answer:
"""
                response = model.generate(prompt, max_tokens=250)
                progress_bar.progress(100)
                status_text.text("Done!")

                end_time = time.time()
                st.markdown("<div class='result-box'><b>📊 Answer:</b></div>", unsafe_allow_html=True)
                for line in response.split("\n"):
                    if line.strip():
                        st.markdown(f"<div class='result-box'>{line.strip()}</div>", unsafe_allow_html=True)

                st.caption(f"⏱ Total time taken: {end_time - start_time:.2f} seconds")
