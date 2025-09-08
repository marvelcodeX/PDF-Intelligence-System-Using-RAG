import fitz  # pymupdf
from sentence_transformers import SentenceTransformer
import faiss
import ollama
import gradio as gr

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
# Step 4: Load PDF on startup
# ----------------------------
pdf_file_path = "AIML T1.pdf"
try:
    chunks = extract_text_from_pdf(pdf_file_path)
    add_to_index(chunks)
    pdf_indexed = True
except Exception as e:
    pdf_indexed = False
    print(f"Error loading or indexing PDF: {str(e)}")

# ----------------------------
# Step 5: Gradio Chatbot with History + UI
def chat_fn(message, history, model_choice):
    if not pdf_indexed:
        return history + [[{"role": "user", "content": message},
                           {"role": "assistant", "content": "⚠️ PDF not indexed. Please check file."}]], ""
    try:
        answer = query_model(message, model_choice)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})
        return history, ""   # clears textbox
    except Exception as e:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"⚠️ Error: {str(e)}"})
        return history, ""

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Row():
        # Left column (small) → model choice + textbox
        with gr.Column(scale=1):
            gr.Markdown("## ⚙️ Settings")

            model_choice = gr.Dropdown(
                choices=["phi3", "qwen2:1.5b", "phi3:mini"],
                value="phi3",
                label="Choose an AI model:"
            )

            if pdf_indexed:
                gr.Markdown(f"✅ Indexed PDF: **{pdf_file_path}** ready to query.")
            else:
                gr.Markdown("⚠️ PDF not indexed successfully; please check error above.")

            msg = gr.Textbox(
                placeholder="Ask a question about the PDF...",
                label="Your Question",
                lines=1   # enter key will submit automatically
            )

        # Right column (large) → chat history
        with gr.Column(scale=3):   # make chat area bigger
            gr.Markdown("## 💬 Chat History")
            history_box = gr.Chatbot(label="Conversation", height=600, type="messages")

            clear = gr.Button("✌🏻 Start New Chat")
            clear.click(lambda: [], None, history_box, queue=False)

    # Link the textbox submit to function
    msg.submit(chat_fn, [msg, history_box, model_choice], [history_box, msg])

demo.launch()
