# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from itertools import cycle

from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever


app = FastAPI(
    title="ai scriptural councellor",
    # ... other args if any
)
#cors configurations
origins = [
    "http://localhost:3000",
    "https://the-mustard-seed.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,           # important if using cookies/auth
    allow_methods=["*"],              # allow GET, POST, PUT, DELETE, OPTIONS, etc.
    allow_headers=["*"],              # allow Content-Type, Authorization, etc.
)

# ──────────────────────────
# CONFIG
# ──────────────────────────
DB_DIR = "chroma_db"
COLLECTION_NAME = "langchain"  # explicit name – helps consistency

# Load Google API keys from environment (comma-separated)
API_KEYS = os.environ.get("GOOGLE_API_KEYS", "").split(",")
if not API_KEYS or API_KEYS == [""]:
    raise ValueError("Set GOOGLE_API_KEYS in environment variables (comma separated)")

key_cycle = cycle(API_KEYS)

def get_embedder():
    """Create Google embedding model with rotating API keys."""
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=next(key_cycle)
    )

GOOGLE_API_KEY = API_KEYS[0]  # First key used for the LLM

# ──────────────────────────
# LOAD VECTORSTORE + EMBEDDER CHECK
# ──────────────────────────
embedder = get_embedder()  # Instantiate once

vectorstore = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embedder,           # Required for query-time embedding
    collection_name=COLLECTION_NAME
)

# Quick startup check: verify embedder works and log dimension
try:
    test_query = "test sentence for dimension check"
    test_embedding = embedder.embed_query(test_query)
    dimension = len(test_embedding)
    print(f"[STARTUP] Embedding model dimension: {dimension} (expected 768 for text-embedding-004)")
except Exception as e:
    print(f"[STARTUP ERROR] Embedder dimension check failed: {str(e)}")

retriever: BaseRetriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

# ──────────────────────────
# PROMPT TEMPLATE
# ──────────────────────────
prompt = ChatPromptTemplate.from_template("""
Use the following pieces of context to answer the question at the end. Own the content above, act like a pastor.
If you don't know the answer, just say that you don't know.
Return:
1. The title(s) of the devotional(s)
2. Date(s) of release
3. Answer to the question
4. Some scriptures for reference
Context: {context}
Question: {question}
""")

# ──────────────────────────
# LLM
# ──────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ──────────────────────────
# API
# ──────────────────────────
class Question(BaseModel):
    question: str

@app.post("/chat")
async def chat(q: Question):
    try:
        result = chain.invoke(q.question)
        return {"answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "RAG Chat API is running!"}


