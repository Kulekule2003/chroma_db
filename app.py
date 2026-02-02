__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import chromadb
from itertools import cycle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ──────────────────────────
# PATH LOGIC (Critical for Docker)
# ──────────────────────────
app = FastAPI(title="Verilia AI - Scriptural RAG")

# Force absolute path to the directory where Docker copies your files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "devo_collection"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────
# API KEYS & EMBEDDINGS
# ──────────────────────────
raw_keys = os.environ.get("GOOGLE_API_KEYS", "")
if not raw_keys:
    # Fallback for local testing if env var isn't set
    API_KEYS = ["YOUR_LOCAL_KEY_HERE"] 
else:
    API_KEYS = [k.strip() for k in raw_keys.split(",") if k.strip()]

key_cycle = cycle(API_KEYS)

def get_embedder():
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=next(key_cycle),
    )

# ──────────────────────────
# VECTOR DATABASE INITIALIZATION
# ──────────────────────────
# PersistentClient ensures we talk to the disk, not memory
chroma_client = chromadb.PersistentClient(path=DB_DIR)

# LangChain wrapper
vectorstore = Chroma(
    client=chroma_client,
    collection_name=COLLECTION_NAME,
    embedding_function=get_embedder(),
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ──────────────────────────
# RAG CHAIN LOGIC
# ──────────────────────────
def format_docs(docs):
    # Safety check: if docs is an error code (int) or empty list
    if not docs or isinstance(docs, int):
        return "No specific devotional context found for this query."
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_template("""
You are a supportive Christian pastor. Use the provided devotional context to answer the question. 
If the context is insufficient, provide a biblically-sound encouraging response.

Context: {context}
Question: {question}
""")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=API_KEYS[0],
    temperature=0.7
)

chain = (
    {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
    | prompt 
    | llm 
    | StrOutputParser()
)

# ──────────────────────────
# ENDPOINTS
# ──────────────────────────
class Question(BaseModel):
    question: str

@app.get("/")
async def root():
    """Health check that lists actual collections found in the DB folder."""
    try:
        collections = chroma_client.list_collections()
        col_names = [c.name for c in collections]
        
        # Try to get count of our specific collection
        count = 0
        if COLLECTION_NAME in col_names:
            count = chroma_client.get_collection(COLLECTION_NAME).count()
            
        return {
            "status": "Online",
            "db_path": DB_DIR,
            "collections_in_db": col_names,
            "target_collection": COLLECTION_NAME,
            "count": count
        }
    except Exception as e:
        return {"status": "Error", "message": str(e)}

@app.post("/chat")
async def chat(q: Question):
    try:
        response = chain.invoke(q.question)
        return {"answer": response}
    except Exception as e:
        # This will show up in Render Logs
        print(f"CRITICAL CHAT ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/files")
async def list_files():
    """Debug route to verify file existence via browser."""
    import os
    return {
        "base_dir": BASE_DIR,
        "db_dir_exists": os.path.exists(DB_DIR),
        "db_files": os.listdir(DB_DIR) if os.path.exists(DB_DIR) else []
    }