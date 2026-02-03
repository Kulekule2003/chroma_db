# app.py
import os
import subprocess
from itertools import cycle

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
# ENV
# ──────────────────────────
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# ──────────────────────────
# PATHS
# ──────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_DIR = os.path.join(BASE_DIR, "chroma_db")
CSV_PATH = os.path.join(BASE_DIR, "devo.csv")
BUILD_SCRIPT = os.path.join(BASE_DIR, "build_db_safe.py")

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
# Comma separated:
# KEY1,KEY2,KEY3,KEY4
# Last key = USER REQUESTS ONLY
# Others = DB BUILD / EMBEDDING
# ──────────────────────────
RAW_KEYS = os.environ.get("GOOGLE_API_KEYS", "")
API_KEYS = [k.strip() for k in RAW_KEYS.split(",") if k.strip()]

if len(API_KEYS) < 1:
    raise RuntimeError("GOOGLE_API_KEYS must contain at least one key")

BUILD_KEYS = API_KEYS[:-1] if len(API_KEYS) > 1 else API_KEYS
USER_KEY = API_KEYS[-1]

build_key_cycle = cycle(BUILD_KEYS)

# ──────────────────────────
# EMBEDDINGS
# ──────────────────────────
def get_embedder():
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=next(build_key_cycle),
    )

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

def rebuild_db_safe():
    """
    Builds DB but NEVER crashes app if API quota is hit
    Uses rotating API keys for embeddings
    """
    print("⚠️ Chroma DB missing — attempting rebuild")

    if not os.path.exists(CSV_PATH):
        print("❌ devo.csv not found — skipping rebuild")
        return False

    if not os.path.exists(BUILD_SCRIPT):
        print("❌ build_db_safe.py not found — skipping rebuild")
        return False

    try:
        subprocess.check_call(["python", BUILD_SCRIPT])
        print("✅ Chroma DB rebuilt successfully")
        return True
    except subprocess.CalledProcessError as e:
        print("⚠️ DB rebuild stopped (likely API quota hit)")
        print(str(e))
        return False

# ──────────────────────────
# STARTUP
# ──────────────────────────
@app.on_event("startup")
def startup():
    global chroma_client, vectorstore, retriever, chain

    print("\n>>>>>>>><<<<< RAG STARTUP")
    print(">>>>>>>><<<<< BASE DIR:", BASE_DIR)
    print(">>>>>>>><<<<< DB PATH:", DB_DIR)
    print(">>>>>>>><<<<< BUILD KEYS:", len(BUILD_KEYS))
    print(">>>>>>>><<<<< USER KEY RESERVED")

    # Build DB if missing
    if not os.path.exists(DB_DIR):
        rebuild_db_safe()

    # Chroma client
    chroma_client = Client(
        Settings(
            persist_directory=DB_DIR,
            anonymized_telemetry=False,
        )
    )

    # Find collections
    collections = chroma_client.list_collections()
    print(">>>>>>>><<<<< Found collections:", [c.name for c in collections])

    if not collections:
        print("⚠️ No collections found — running in EMPTY DB MODE")
        collection_name = None
    else:
        collection_name = collections[0].name
        print(">>>>>>>><<<<< Using collection:", collection_name)

    # Vectorstore setup
    embedder = get_embedder()

    try:
        test_embedding = embedder.embed_query("dimension test")
        print(f"[STARTUP] Embedding dimension: {len(test_embedding)}")
    except Exception as e:
        print("⚠️ Embedder failed:", str(e))

    if collection_name:
        vectorstore = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=embedder,
        )

        doc_count = chroma_client.get_collection(collection_name).count()
        print(">>>>>>>><<<<< Documents in DB:", doc_count)
    else:
        vectorstore = None
        doc_count = 0

    # Retriever
    if vectorstore and doc_count > 0:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2},
        )
    else:
        retriever = None

    # LLM — USE RESERVED USER KEY ONLY
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=USER_KEY,
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
    def safe_retrieve(q):
        if not retriever:
            return "Devotional database is unavailable or empty."
        return retriever.invoke(q)

    chain = (
        {
            "context": RunnableLambda(safe_retrieve) | RunnableLambda(format_docs),
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
    return {
        "status": "RAG Chat API running",
        "build_keys": len(BUILD_KEYS),
        "user_key_reserved": True,
    }

@app.post("/chat")
async def chat(q: Question):
    try:
        result = chain.invoke(q.question)
        return {"answer": result}
    except Exception as e:
        print("[CHAT ERROR]", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug-db")
async def debug_db():
    try:
        collections = chroma_client.list_collections()
        return {
            "db_path": DB_DIR,
            "collections": [
                {
                    "name": c.name,
                    "documents": chroma_client.get_collection(c.name).count(),
                }
                for c in collections
            ],
        }
    except Exception as e:
        return {"error": str(e)}
