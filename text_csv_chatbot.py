import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from gpt4all import GPT4All
import re

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

# -------------------------------
# CSV Handling
# -------------------------------
def query_csv(question, df, model, max_tokens=250):
    sample = df.head(10).to_csv(index=False)
    cols = ", ".join(df.columns)
    context = f"Columns: {cols}\n\nSample data:\n{sample}"
    context = context[:1500]

    prompt = f"""
You are a data analyst. Use the CSV structure and sample below to answer questions.
If the answer cannot be determined from the CSV, respond: 'The answer is not present in the CSV.'

{context}

Question: {question}
Answer:
"""
    response = model.generate(prompt, max_tokens=max_tokens)
    return response

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="PDF & CSV QnA", layout="wide")
st.title("📘 PDF & 📊 CSV Q&A Assistant")

embedder = load_embedder()
model = load_gpt4all()

tab_pdf, tab_csv = st.tabs(["📘 PDF Q&A", "📊 CSV Q&A"])

# ---- PDF Tab ----
with tab_pdf:
    st.header("Upload PDF")
    pdf_file = st.file_uploader("Choose a PDF", type="pdf")

    if pdf_file:
        chunks = extract_pdf_chunks(pdf_file)
        chunk_embeddings = np.array(embedder.encode(chunks), dtype='float32')
        st.success(f"✅ PDF processed into {len(chunks)} chunks.")

        question = st.text_input("Ask a question about the PDF:")
        if question:
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Step 1: Encode question
            status_text.text("Encoding question...")
            q_emb = np.array(embedder.encode([question]), dtype='float32')
            progress_bar.progress(20)

            # Step 2: Compute cosine similarity with all chunks
            status_text.text("Computing similarity with PDF chunks...")
            sims = cos_sim(q_emb, chunk_embeddings)[0].cpu().numpy()
            progress_bar.progress(50)

            # Step 3: Filter by similarity threshold
            SIM_THRESHOLD = 0.4
            relevant_idxs = np.where(sims >= SIM_THRESHOLD)[0]
            if len(relevant_idxs) == 0:
                progress_bar.progress(100)
                status_text.text("No relevant chunks found.")
                st.markdown("### 📖 Answer:")
                st.write("The answer is not present in the PDF.")
            else:
                relevant_chunks = [chunks[i] for i in relevant_idxs]
                status_text.text("Highlighting most relevant sentences in top chunks...")

                # Highlight the most relevant sentence in each chunk
                for rank, idx in enumerate(relevant_idxs, start=1):
                    chunk_text = chunks[idx]
                    sentences = re.split(r'(?<=[.!?]) +', chunk_text)
                    sent_embeddings = np.array(embedder.encode(sentences), dtype='float32')
                    sent_sims = cos_sim(q_emb, sent_embeddings)[0].cpu().numpy()
                    max_idx = np.argmax(sent_sims)
                    highlighted_sentence = sentences[max_idx]

                    display_text = chunk_text.replace(
                        highlighted_sentence,
                        f"**{highlighted_sentence}**"
                    )
                    st.markdown(f"**Chunk {rank} (similarity: {sims[idx]:.2f}):**")
                    st.write(display_text[:500] + ("..." if len(display_text) > 500 else ""))
                progress_bar.progress(80)

                # Step 4: Generate answer using only relevant chunks
                status_text.text("Generating answer from the model...")
                context = " ".join(relevant_chunks)[:3000]
                prompt = f"""
You are a strict assistant. Answer the question based ONLY on the PDF content below.
If the answer is not present in the PDF, respond: 'The answer is not present in the PDF.'

Context:
{context}

Question: {question}
Answer:
"""
                response = model.generate(prompt, max_tokens=200)
                progress_bar.progress(100)
                status_text.text("Done!")

                st.markdown("### 📖 Answer:")
                st.write(response)

# ---- CSV Tab ----
with tab_csv:
    st.header("Upload CSV")
    csv_file = st.file_uploader("Choose a CSV", type="csv")

    if csv_file:
        df = pd.read_csv(csv_file)
        st.success(f"✅ CSV loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
        st.dataframe(df.head(10))

        question = st.text_input("Ask a question about the CSV:")
        if question:
            with st.spinner("Analyzing..."):
                answer = query_csv(question, df, model)
            st.markdown("### 📊 Answer:")
            st.write(answer)
