# app.py
import os
from itertools import cycle
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever

from langchain_chroma import Chroma

from chromadb import Client
from chromadb.config import Settings

# ──────────────────────────
# ENV / TELEMETRY
# ──────────────────────────
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# ──────────────────────────
# PATH RESOLUTION
# Works locally + Render
# ──────────────────────────
BASE_DIR = os.getenv("RENDER_PROJECT_DIR", os.getcwd())
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# ──────────────────────────
# APP
# ──────────────────────────
app = FastAPI(title="AI Scriptural Counsellor")

# ──────────────────────────
# CORS
# ──────────────────────────
origins = [
    "http://localhost:3000",
    "https://the-mustard-seed.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────
# GOOGLE API KEYS
# ──────────────────────────
API_KEYS = os.environ.get("GOOGLE_API_KEYS", "").split(",")
if not API_KEYS or API_KEYS == [""]:
    raise RuntimeError("GOOGLE_API_KEYS environment variable not set")

key_cycle = cycle(API_KEYS)

# ──────────────────────────
# EMBEDDINGS
# ──────────────────────────
def get_embedder():
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=next(key_cycle),
    )

GOOGLE_API_KEY = API_KEYS[0]

# ──────────────────────────
# HELPERS
# ──────────────────────────
def format_docs(docs):
    if not docs:
        return "No devotional context found in database."

    return "\n\n".join(
        f"Title: {doc.metadata.get('title', 'Unknown')}\n"
        f"Date: {doc.metadata.get('date', 'Unknown')}\n"
        f"Content:\n{doc.page_content}"
        for doc in docs
    )

def fail_if_missing_db():
    if not os.path.exists(DB_DIR):
        print("❌ chroma_db directory NOT FOUND at:", DB_DIR)
        raise RuntimeError(
            "Chroma DB folder missing. "
            "You must either:\n"
            "1) Use Render Persistent Disk\n"
            "OR\n"
            "2) Rebuild the DB at startup\n"
        )

# ──────────────────────────
# STARTUP
# ──────────────────────────
@app.on_event("startup")
def startup():
    global chroma_client, vectorstore, retriever, chain

    print("\n>>>>>>>><<<<< RAG STARTUP")
    print(">>>>>>>><<<<< BASE DIR:", BASE_DIR)
    print(">>>>>>>><<<<< DB PATH:", DB_DIR)

    # HARD FAIL if DB missing
    fail_if_missing_db()

    # Chroma client
    chroma_client = Client(
        Settings(
            persist_directory=DB_DIR,
            anonymized_telemetry=False,
        )
    )

    # List collections ON DISK
    collections = chroma_client.list_collections()
    print(">>>>>>>><<<<< Found collections:", [c.name for c in collections])

    if not collections:
        raise RuntimeError("No Chroma collections found in DB")

    # Use first collection automatically
    collection_name = collections[0].name
    print(">>>>>>>><<<<< Using collection:", collection_name)

    # Embeddings
    embedder = get_embedder()

    # Dimension test
    try:
        test_embedding = embedder.embed_query("dimension test")
        print(
            f"[STARTUP] Embedding dimension: {len(test_embedding)} "
            "(expected 768)"
        )
    except Exception as e:
        raise RuntimeError(f"Embedding model failed: {str(e)}")

    # Vectorstore
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embedder,
    )

    # Verify DB has documents
    doc_count = chroma_client.get_collection(collection_name).count()
    print(">>>>>>>><<<<< Documents in DB:", doc_count)

    if doc_count == 0:
        raise RuntimeError("Chroma collection is EMPTY — RAG will not work")

    # Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2},
    )

    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
    )

    # Prompt
    prompt = ChatPromptTemplate.from_template(
        """
You are a Christian pastor and devotional guide.

Use the following context to answer the question.
If the context does not contain the answer, say you don't know.

Return:
1. Title(s) of devotional(s)
2. Date(s) of release
3. Answer to the question
4. Relevant scripture references

Context:
{context}

Question:
{question}
"""
    )

    # Chain
    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print(">>>>>>>><<<<< RAG READY\n")

# ──────────────────────────
# API MODELS
# ──────────────────────────
class Question(BaseModel):
    question: str

# ──────────────────────────
# ROUTES
# ──────────────────────────
@app.get("/")
async def root():
    return {"status": "RAG Chat API running"}

@app.post("/chat")
async def chat(q: Question):
    try:
        result = chain.invoke(q.question)
        return {"answer": result}
    except Exception as e:
        print("[CHAT ERROR]", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# ──────────────────────────
# DEBUG ROUTES
# ──────────────────────────
@app.get("/debug-db")
async def debug_db():
    try:
        collections = chroma_client.list_collections()
        data = []

        for c in collections:
            col = chroma_client.get_collection(c.name)
            data.append({
                "name": c.name,
                "documents": col.count()
            })

        return {
            "db_path": DB_DIR,
            "collections": data
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug-fs")
async def debug_fs():
    try:
        return {
            "cwd": os.getcwd(),
            "base_dir": BASE_DIR,
            "db_path": DB_DIR,
            "db_exists": os.path.exists(DB_DIR),
            "root_files": os.listdir(BASE_DIR),
            "db_files": os.listdir(DB_DIR) if os.path.exists(DB_DIR) else [],
        }
    except Exception as e:
        return {"error": str(e)}
