import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from gpt4all import GPT4All
from sympy import false


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
# Extract PDF text
# ---------------------------
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    return " ".join([page.extract_text() or "" for page in reader.pages])


# ---------------------------
# Dynamically determine chunk size
# ---------------------------
def determine_chunk_size(text, min_size=150, max_size=400):
    total_words = len(text.split())
    if total_words < 1000:
        return min_size
    elif total_words > 5000:
        return max_size
    else:
        # interpolate linearly between min_size and max_size
        return min_size + int((total_words - 1000) / 4000 * (max_size - min_size))


# ---------------------------
# Split text into chunks
# ---------------------------
def split_chunks(text, overlap=50):
    chunk_size = determine_chunk_size(text)
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


# ---------------------------
# Build FAISS index
# ---------------------------
@st.cache_resource
def build_index(chunks):
    embedder = load_embedder()
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))
    return index, embeddings


# ---------------------------
# Retrieve relevant chunks
# ---------------------------
def search(query, chunks, index, top_k=3, max_context_tokens=500):
    embedder = load_embedder()
    q_vec = embedder.encode([query])
    _, idx = index.search(np.array(q_vec).astype("float32"), top_k)

    selected_chunks = []
    token_count = 0
    for i in idx[0]:
        chunk_tokens = len(chunks[i].split())
        if token_count + chunk_tokens > max_context_tokens:
            break
        selected_chunks.append(chunks[i])
        token_count += chunk_tokens
    return selected_chunks


# ---------------------------
# Streamlit app
# ---------------------------
st.title("📄 PDF Q&A (Context-Strict, Dynamic Chunks)")

uploaded = st.file_uploader("Upload PDF", type="pdf")
if uploaded:
    text = extract_text_from_pdf(uploaded)
    st.success("✅ PDF loaded successfully!")

    chunks = split_chunks(text)
    st.info(f"Split PDF into {len(chunks)} dynamic chunks.")
    index, _ = build_index(chunks)

    query = st.text_input("Ask a question:")
    if query:
        relevant = search(query, chunks, index, top_k=3, max_context_tokens=500)
        if not relevant:
            st.warning("No relevant context found in PDF.")
        else:
            context = "\n".join(relevant)
            gpt = load_gpt4all()

            prompt = f"""
You are a helpful assistant.
Use ONLY the following context to answer the question.
If the answer is not in the context, reply exactly: "I don't know".

Context:
{context}

Question: {query}
Answer:
"""

            with gpt.chat_session():
                ans = gpt.generate(prompt, max_tokens=150)

            st.markdown("### 💡 Answer")
            st.write(ans.strip())
