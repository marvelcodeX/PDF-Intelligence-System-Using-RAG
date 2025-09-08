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
    prompt = (
        f"Answer the question using only the context below.\n\nContext:\n{context}\n\n"
        f"Question: {query}\n\nAnswer with sources."
    )
    response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# ----------------------------
# Step 4: Streamlit UI
# ----------------------------
st.set_page_config(page_title="🎀 PDF Intelligence System 🎀", layout="wide")

# Initialize session state for chat history and current answer, if not present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_answer" not in st.session_state:
    st.session_state.current_answer = ""

# Load and index PDF once
pdf_file_path = "AIML T1.pdf"
try:
    chunks = extract_text_from_pdf(pdf_file_path)
    add_to_index(chunks)
    pdf_indexed = True
except Exception as e:
    pdf_indexed = False
    st.error(f"Error loading or indexing PDF: {str(e)}")

# Sidebar for chat history and new chat button
with st.sidebar:
    st.header("💬 Chat History")
    if st.session_state.chat_history:
        for i, chat in enumerate(st.session_state.chat_history[::-1]):
            st.markdown(f"**Q:** {chat['question']}")
            if i < len(st.session_state.chat_history) - 1:
                st.markdown("---")
    else:
        st.info("No chat history yet.")
    st.markdown("---")
    if st.button("✌🏻 Start New Chat"):
        st.session_state.chat_history = []
        st.session_state.current_answer = ""
        # Optional: clear query input by rerunning app - can be handled on main page

# Main content
st.title("🎀 PDF Intelligence System (Free RAG) 🎀")
st.markdown("---")

# Model selector on top
model_choice = st.selectbox(
    "Choose an AI model:",
    options=["phi3", "qwen2:1.5b", "phi3:mini"],
    index=0,
)

# Show PDF indexing status
if pdf_indexed:
    st.info(f"Indexed PDF: **{pdf_file_path}** ✅ Processed and ready to query.")
else:
    st.warning("PDF not indexed successfully; please check the error above.")

st.markdown("---")

# Query input area
query = st.text_area(
    label="Ask a question about the PDF document:",
    placeholder="Type your question here...",
    height=120,
    max_chars=500,
    key="query_input"  # allows clearing on new chat if needed
)

st.markdown("---")

if st.button("Get Answer") and query.strip():
    try:
        with st.spinner("Generating answer..."):
            answer = query_model(query, model_choice)
        st.session_state.current_answer = answer
        # Save Q&A to chat history
        st.session_state.chat_history.append({"question": query, "answer": answer})

        # Display answer in a styled box with a copy button
        st.markdown(
            f"""
            <div style="
                background-color: #f9f9fb;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 7px rgba(0, 0, 0, 0.1);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                white-space: pre-wrap;
                color: #222;
                position: relative;
            ">
                <h3 style="color:#333;">Answer:</h3>
                <p id="answer-text">{answer}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        import traceback
        st.text(traceback.format_exc())
elif not query.strip():
    st.info("Please enter a question to get an answer from the PDF.")