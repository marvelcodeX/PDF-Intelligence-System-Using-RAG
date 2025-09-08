# PDF-Intelligence-System-Using-RAG
A lightweight **Retrieval-Augmented Generation (RAG)** system that lets you query PDFs using **local LLMs with Ollama**.   Built with **Streamlit**, **FAISS**, **Sentence Transformers**, and **Ollama**.  

✅ 100% local, no API keys  
✅ Choose from small, free models (Phi-3, qwen2:1.5b, phi3:mini, etc.)  
✅ Ask natural language questions about any PDF  

---

## ⚡ Features
- Parse and index any PDF with [PyMuPDF](https://pymupdf.readthedocs.io/)  
- Embed text chunks using [Sentence Transformers](https://www.sbert.net/)  
- Store & search embeddings with [FAISS](https://github.com/facebookresearch/faiss)  
- Query using local LLMs via [Ollama](https://ollama.ai)  
- Simple [Streamlit](https://streamlit.io/) web UI
- Alternate: [Gradio](https://www.gradio.app/) UI
---

## Models

This repo is designed to work with lightweight models (<2GB).

Pull them with Ollama:

```bash
ollama pull phi3
ollama pull phi3 mini
ollama pull qwen2:1.5b
```

(You can replace with other models like phi3:mini or tinyllama:1.1b if you want smaller ones.)

---

## Run the streamlit app

```bash
streamlit run app.py
```

---

## Run the Gradio app

```bash
python app1.py
```

