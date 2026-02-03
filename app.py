# app.py
import os
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
# ENV / TELEMETRY
# ──────────────────────────
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# ──────────────────────────
# PATH RESOLUTION
# Works locally and on Render
# ──────────────────────────
BASE_DIR = os.getenv("RENDER_PROJECT_DIR", os.getcwd())
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

COLLECTION_NAME = "langchain"

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
# CHROMA CLIENT (VERSION SAFE)
# ──────────────────────────
chroma_client = Client(
    Settings(
        persist_directory=DB_DIR,
        anonymized_telemetry=False,
    )
)

# ──────────────────────────
# VECTORSTORE
# ──────────────────────────
embedder = get_embedder()

vectorstore = Chroma(
    client=chroma_client,
    collection_name=COLLECTION_NAME,
    embedding_function=embedder,
)

print(">>>>>>>><<<<< Loaded collection:", COLLECTION_NAME)
print(">>>>>>>><<<<< DB PATH:", DB_DIR)

# Startup sanity check
try:
    test_embedding = embedder.embed_query("dimension test")
    print(
        f"[STARTUP] Embedding dimension: {len(test_embedding)} "
        f"(expected 768 for text-embedding-004)"
    )
except Exception as e:
    print("[STARTUP ERROR] Embedder failed:", str(e))

# ──────────────────────────
# RETRIEVER
# ──────────────────────────
retriever: BaseRetriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2},
)

# ──────────────────────────
# CONTEXT FORMATTER
# ──────────────────────────
def format_docs(docs):
    if not docs:
        return "No devotional context found."

    return "\n\n".join(
        f"Title: {doc.metadata.get('title', 'Unknown')}\n"
        f"Date: {doc.metadata.get('date', 'Unknown')}\n"
        f"Content:\n{doc.page_content}"
        for doc in docs
    )

# ──────────────────────────
# PROMPT
# ──────────────────────────
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

# ──────────────────────────
# LLM
# ──────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
)

# ──────────────────────────
# CHAIN
# ──────────────────────────
chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

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
        raise HTTPException(status_code=500, detail="RAG processing failed")

# ──────────────────────────
# DEBUG: DATABASE
# ──────────────────────────
@app.get("/debug-db")
async def debug_db():
    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
        count = collection.count()

        return {
            "collection_name": COLLECTION_NAME,
            "documents_in_db": count,
            "db_path": DB_DIR,
        }
    except Exception as e:
        return {"error": str(e)}

# ──────────────────────────
# DEBUG: FILESYSTEM
# ──────────────────────────
@app.get("/debug-fs")
async def debug_fs():
    try:
        cwd = os.getcwd()
        root_files = os.listdir(cwd)

        chroma_exists = os.path.exists(DB_DIR)
        chroma_files = []
        if chroma_exists:
            chroma_files = os.listdir(DB_DIR)

        return {
            "cwd": cwd,
            "base_dir": BASE_DIR,
            "db_path": DB_DIR,
            "root_files": root_files,
            "chroma_db_exists": chroma_exists,
            "chroma_db_files": chroma_files,
        }
    except Exception as e:
        return {"error": str(e)}
