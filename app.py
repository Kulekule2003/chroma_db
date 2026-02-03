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
# CONFIG & PATHS
# ──────────────────────────
os.environ["ANONYMIZED_TELEMETRY"] = "False"
app = FastAPI(title="AI Scriptural Counsellor")

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "devo_collection"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────
# API KEYS
# ──────────────────────────
raw_keys = os.environ.get("GOOGLE_API_KEYS", "")
if not raw_keys:
    raise RuntimeError("GOOGLE_API_KEYS environment variable is missing")

API_KEYS = [k.strip() for k in raw_keys.split(",") if k.strip()]
key_cycle = cycle(API_KEYS)

def get_embedder():
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=next(key_cycle),
    )

# ──────────────────────────
# VECTORSTORE & RETRIEVER
# ──────────────────────────
chroma_client = chromadb.PersistentClient(path=DB_DIR)

vectorstore = Chroma(
    client=chroma_client,
    collection_name=COLLECTION_NAME,
    embedding_function=get_embedder(),
)

# Set up retriever explicitly
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ──────────────────────────
# THE FIX: SAFE FORMATTING
# ──────────────────────────
def format_docs(docs):
    # This check prevents the 'int' has no len() error
    if not docs or isinstance(docs, int):
        return "No relevant devotional content found."
    
    # Ensure we are iterating over a list of documents
    return "\n\n".join(doc.page_content for doc in docs)

# ──────────────────────────
# PROMPT & LLM
# ──────────────────────────
prompt = ChatPromptTemplate.from_template("""
You are a Christian pastor. Use the following snippets from devotional writings to answer the question. 
If the context doesn't contain the answer, use your knowledge of the Bible to provide a supportive, pastoral response.

Context: {context}
Question: {question}
""")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=API_KEYS[0],
    temperature=0.7
)

# Build the chain with the fixed formatter
chain = (
    {
        "context": retriever | RunnableLambda(format_docs), 
        "question": RunnablePassthrough()
    }
    | prompt 
    | llm 
    | StrOutputParser()
)

# ──────────────────────────
# ROUTES
# ──────────────────────────
class Question(BaseModel):
    question: str

@app.get("/")
async def root():
    try:
        count = chroma_client.get_collection(COLLECTION_NAME).count()
        return {"status": "RAG API Online", "docs_count": count}
    except:
        return {"status": "Online", "error": "Collection not found"}

@app.post("/chat")
async def chat(q: Question):
    try:
        # Debugging: check if question is received
        if not q.question:
            raise HTTPException(status_code=400, detail="Question is empty")
            
        response = chain.invoke(q.question)
        return {"answer": response}
    except Exception as e:
        # Print the full error to Render logs
        print(f"CHAT ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/debug/ls")
def list_files(path: str = "."):
    # Returns a list of files in the specified directory
    try:
        files = os.listdir(path)
        return {"directory": path, "content": files}
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug/env")
def show_env():
    # Shows environment variables (Careful: hides secrets!)
    return {k: v for k, v in os.environ.items() if "SECRET" not in k and "PASSWORD" not in k}