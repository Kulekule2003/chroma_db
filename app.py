__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import chromadb
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

# ──────────────────────────
# CONFIG & PATHS
# ──────────────────────────
os.environ["ANONYMIZED_TELEMETRY"] = "False"
app = FastAPI(title="AI Scriptural Counsellor")

# Find chroma_db folder relative to this file
current_dir = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(current_dir, "chroma_db")
COLLECTION_NAME = "devo_collection"

# ──────────────────────────
# CORS
# ──────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust for production security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────
# EMBEDDINGS & KEYS
# ──────────────────────────
API_KEYS = os.environ.get("GOOGLE_API_KEYS", "").split(",")
if not API_KEYS or API_KEYS == [""]:
    raise RuntimeError("GOOGLE_API_KEYS environment variable not set")

key_cycle = cycle(API_KEYS)

def get_embedder():
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=next(key_cycle),
    )

# ──────────────────────────
# VECTORSTORE INITIALIZATION
# ──────────────────────────
# Initialize modern persistent client
chroma_client = chromadb.PersistentClient(path=DB_DIR)

vectorstore = Chroma(
    client=chroma_client,
    collection_name=COLLECTION_NAME,
    embedding_function=get_embedder(),
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# ──────────────────────────
# RAG CHAIN
# ──────────────────────────
def format_docs(docs):
    if not docs: return "No context found."
    return "\n\n".join(
        f"Title: {d.metadata.get('title', 'Unknown')}\nContent: {d.page_content}"
        for d in docs
    )

prompt = ChatPromptTemplate.from_template("""
You are a Christian pastor. Use the context to answer the question.
Context: {context}
Question: {question}
""")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", # Updated to current stable model
    google_api_key=API_KEYS[0],
)

chain = (
    {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

# ──────────────────────────
# ROUTES
# ──────────────────────────
class Question(BaseModel):
    question: str

@app.get("/")
async def root():
    return {"status": "RAG Chat API running"}

@app.post("/chat")
async def chat(q: Question):
    try:
        result = chain.invoke(q.question)
        return {"answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug-db")
async def debug_db():
    try:
        count = chroma_client.get_collection(COLLECTION_NAME).count()
        return {"collection": COLLECTION_NAME, "count": count, "path": DB_DIR}
    except Exception as e:
        return {"error": str(e)}