import streamlit as st
import fitz  # pymupdf
from sentence_transformers import SentenceTransformer
import faiss
import ollama

# ----------------------------
# Step 1: PDF Parsing
# ----------------------------
def extract_text_from_pdf(pdf_file_path):
    doc = fitz.open(pdf_file_path)
    text_chunks = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            text_chunks.append({
                "content": text,
                "source": f"{pdf_file_path} - Page {page_num+1}"
            })
    return text_chunks

# ----------------------------
# Step 2: Embeddings + FAISS
# ----------------------------
embedder = SentenceTransformer("BAAI/bge-small-en")
dimension = 384  # embedding size for bge-small
index = faiss.IndexFlatL2(dimension)
docs = []

def add_to_index(chunks):
    global docs
    vectors = embedder.encode([c["content"] for c in chunks])
    index.add(vectors)
    docs.extend(chunks)

def search(query, k=3):
    q_vec = embedder.encode([query])
    distances, indices = index.search(q_vec, k)
    results = [docs[i] for i in indices[0]]
    return results

# ----------------------------
# Step 3: Query with chosen model
# ----------------------------
def query_model(query, model_name):
    results = search(query)
    context = "\n\n".join([f"{r['content']} (Source: {r['source']})" for r in results])
    prompt = f"Answer the question using only the context below.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer with sources."
    response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# ----------------------------
# Step 4: Streamlit UI
# ----------------------------
st.title("📄 PDF Intelligence System (Free RAG)")

# Hardcode your PDF file path here (update for Mac path!)
pdf_file_path = "AIML T1.pdf"

# Extract and index
chunks = extract_text_from_pdf(pdf_file_path)
add_to_index(chunks)
st.success("Hardcoded PDF processed and indexed ✅")

query = st.text_input("Ask a question:")

# ✅ Use phi3 + qwen2:1.5b + gemma:2b
model_choice = st.selectbox("Choose a model:", [
    "phi3",
    "qwen2:1.5b",
    "gemma:2b"
])

if st.button("Get Answer") and query:
    answer = query_model(query, model_choice)
    st.write("### Answer:")
    st.write(answer)
