# 🚀 RAG-based Compliance Intelligence System with FastAPI Backend

An AI-powered compliance assistant that allows users to upload legal/compliance documents and ask questions. The system uses Retrieval-Augmented Generation (RAG) to provide accurate, context-based answers along with risk insights.

---

## 🧠 Features

- 📄 Upload PDF documents
- 💬 Ask questions based on document
- ⚡ Semantic search using FAISS
- 🤖 AI-generated answers using LLM
- ⚠️ Risk detection (penalty, violation, deadline)
- 📊 Document summarization
- 🔍 Explainability with source chunks

---

## 🏗️ Tech Stack

- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Embeddings:** Sentence Transformers (MiniLM)
- **Vector DB:** FAISS
- **LLM:** TinyLlama (Ollama)

---

## ⚙️ How It Works

1. Upload document
2. Text is extracted and chunked
3. Embeddings are generated
4. Stored in FAISS vector DB
5. Query → retrieve relevant chunks
6. LLM generates answer using context

---

## 🚀 Run Locally

### 1. Clone repo
```bash
git clone <your-repo-link>
cd rag-compliance-assistant