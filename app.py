import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# ==============================
# CONFIG
# ==============================

BASE_DIR = os.getcwd()

DB_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "langchain"

EMBEDDING_MODEL = "models/text-embedding-004"
LLM_MODEL = "gemini-2.5-flash"

EXPECTED_DIM = 768

# ==============================
# FASTAPI
# ==============================

app = FastAPI()

# ==============================
# MODELS
# ==============================

class QueryRequest(BaseModel):
    question: str

# ==============================
# STARTUP
# ==============================

@app.on_event("startup")
def startup():
    global vectordb, retriever, llm

    print(">>>>>>>><<<<< DB PATH:", DB_DIR)

    if not os.path.exists(DB_DIR):
        print("❌ chroma_db folder NOT FOUND")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    vectordb = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    count = vectordb._collection.count()

    print(">>>>>>>><<<<< Loaded collection:", COLLECTION_NAME)
    print(">>>>>>>><<<<< Documents in DB:", count)

    if count == 0:
        print("❌ WARNING: DATABASE IS EMPTY — CONTEXT WILL NOT WORK")

    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0.2
    )

    dim = len(embeddings.embed_query("test"))
    print(f"[STARTUP] Embedding dimension: {dim} (expected {EXPECTED_DIM})")


# ==============================
# ROUTES
# ==============================

@app.get("/")
def root():
    return {"status": "RAG API Running"}

@app.get("/debug-db")
def debug_db():
    exists = os.path.exists(DB_DIR)

    if not exists:
        return {
            "db_path": DB_DIR,
            "exists": False,
            "documents_in_db": 0
        }

    count = vectordb._collection.count()

    return {
        "db_path": DB_DIR,
        "exists": True,
        "collection_name": COLLECTION_NAME,
        "documents_in_db": count
    }


@app.post("/ask")
def ask(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")

    docs = retriever.get_relevant_documents(req.question)

    if not docs:
        return {
            "answer": "No relevant context found in database.",
            "sources": []
        }

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are a devotional assistant.
Answer ONLY using the context below.

Context:
{context}

Question:
{req.question}
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": len(docs)
    }
