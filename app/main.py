from fastapi import FastAPI, UploadFile, File
import shutil
import os
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama

app = FastAPI()

vectorstore = None
retriever = None


def detect_risks(text):
    risk_keywords = ["penalty", "fine", "violation", "deadline", "audit"]
    found_risks = []

    for word in risk_keywords:
        if word.lower() in text.lower():
            found_risks.append(word)

    return found_risks


@app.get("/")
def home():
    return {"message": "AI Compliance Assistant Running 🚀"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vectorstore, retriever

    file_path = f"data/{file.filename}"
    os.makedirs("data", exist_ok=True)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    return {
        "message": "File uploaded and processed successfully",
        "pages_loaded": len(docs),
        "chunks_created": len(chunks)
    }


@app.post("/ask")
def ask_question(query: str):
    global retriever

    if retriever is None:
        return {"error": "Please upload a document first"}

    start = time.time()

    retrieved_docs = retriever.invoke(query)

    if not retrieved_docs:
        return {"answer": "No relevant information found"}

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

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

    risks = detect_risks(context)

    end = time.time()

    return {
        "answer": answer,
        "risks_detected": risks,
        "response_time": round(end - start, 2),
        "source_chunks": [doc.page_content[:200] for doc in retrieved_docs]
    }


@app.get("/summary")
def summarize_document():
    global vectorstore

    if vectorstore is None:
        return {"error": "No document uploaded yet."}

    start = time.time()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    docs = retriever.invoke("Summarize the important points of this document")

    context = "\n\n".join([doc.page_content for doc in docs])

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

    end = time.time()

    return {
        "summary": response.content,
        "response_time": round(end - start, 2)
    }