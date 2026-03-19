from fastapi import FastAPI, UploadFile, File
import shutil
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama

app = FastAPI()

# Global variables
vectorstore = None
retriever = None

# 🔥 Risk Detection Function
def detect_risks(text):
    risk_keywords = ["penalty", "fine", "violation", "deadline", "audit"]
    found_risks = []

    for word in risk_keywords:
        if word.lower() in text.lower():
            found_risks.append(word)

    return found_risks


# 🏠 Home route
@app.get("/")
def home():
    return {"message": "AI Compliance Assistant Running 🚀"}


# 📄 Upload PDF
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vectorstore, retriever

    # Save file
    file_path = f"data/{file.filename}"
    os.makedirs("data", exist_ok=True)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Vector DB
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    return {
        "message": "File uploaded and processed successfully",
        "pages_loaded": len(docs),
        "chunks_created": len(chunks)
    }


# ❓ Ask Question
@app.post("/ask")
def ask_question(query: str):
    global retriever

    if retriever is None:
        return {"error": "Please upload a document first"}

    # Retrieve docs
    retrieved_docs = retriever.invoke(query)

    if not retrieved_docs:
        return {"answer": "No relevant information found"}

    # Combine context
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # LLM (TinyLlama - works on low RAM)
    llm = ChatOllama(model="tinyllama")

    prompt = f"""
You are a strict AI Compliance Assistant.

Rules:
- Answer ONLY using the given context
- If answer not found, say "Not found in document"
- Keep answer clear and structured
- Do NOT add extra knowledge

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)
    answer = response.content

    # 🔥 Risk detection
    risks = detect_risks(context)

    return {
        "answer": answer,
        "risks_detected": risks,
        "source_chunks": [doc.page_content[:200] for doc in retrieved_docs]
    }
    
@app.get("/summary")
def summarize_document():
    global vectorstore

    if vectorstore is None:
        return {"error": "No document uploaded yet."}

    # 🔥 Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    docs = retriever.invoke("Summarize the important points of this document")

    context = "\n\n".join([doc.page_content for doc in docs])

    # 🔥 ADD THIS (missing part)
    llm = ChatOllama(model="tinyllama")

    prompt = f"""
You are a compliance expert.

Tasks:
1. Summarize the document clearly
2. Highlight important compliance points
3. Mention any risks (penalty, violation, deadline)

Context:
{context}

Output format:
Summary:
- ...

Risks:
- ...
"""

    response = llm.invoke(prompt)

    return {
        "summary": response.content
    }