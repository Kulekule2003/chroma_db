from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging

# Disable ChromaDB GPU warnings
logging.getLogger("chromadb").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

app = FastAPI(title="Verilia Devotional RAG API", version="1.0.0")

# CRITICAL: Render port binding confirmation
PORT = os.getenv("PORT", "10000")
print(f"🚀 Starting Verilia RAG API on PORT: {PORT}")
print(f"✅ Host: 0.0.0.0")

# API Key validation
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY environment variable is required")

# Initialize FREE embeddings (no quota usage)
print("🔄 Loading FREE HuggingFace embeddings...")
try:
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("✅ Embeddings loaded successfully")
except Exception as e:
    print(f"❌ Embeddings failed: {e}")
    embedding_model = None

# Initialize Chroma vector store
print("🔄 Loading Chroma vector store...")
try:
    vector_store = Chroma(
        persist_directory="./chroma_db_free",
        embedding_function=embedding_model
    )
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    print("✅ Chroma DB loaded successfully")
except Exception as e:
    print(f"❌ Chroma DB failed: {e}")
    retriever = None

# Initialize LLM (Gemini - quota applies only here)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.1
)

# RAG Prompt Template
prompt = ChatPromptTemplate.from_template("""
You are a pastor providing devotional guidance. Use ONLY the context below to answer.

Answer confidently like you're preaching. Structure your response:
1. **Title(s)** of devotionals used
2. **Date(s)** mentioned
3. **Answer** to the question  
4. **Scripture references** (if relevant)

If context doesn't contain the answer, say: "I don't have specific guidance on this topic in our devotionals."

CONTEXT:
{context}

QUESTION: {question}

RESPONSE:
""")

# RAG Chain (lazy initialization)
chain = None
if retriever and embedding_model:
    try:
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        print("✅ RAG chain ready!")
    except Exception as e:
        print(f"❌ RAG chain failed: {e}")
        chain = None

# Pydantic models
class QueryRequest(BaseModel):
    question: str

class HealthResponse(BaseModel):
    status: str
    rag_ready: bool
    port: str

# ROUTES - Render port detection FIRST
@app.get("/", tags=["health"])
async def root():
    """Render port detection endpoint"""
    return {
        "message": "Verilia Devotional RAG API LIVE! 🚀",
        "endpoints": ["/health", "/query"],
        "port": PORT
    }

@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Detailed health check"""
    rag_ready = chain is not None
    return HealthResponse(
        status="healthy" if rag_ready else "degraded",
        rag_ready=rag_ready,
        port=PORT
    )

@app.post("/query", tags=["rag"])
async def query_devotional(request: QueryRequest):
    """Main RAG endpoint for devotional questions"""
    if not chain:
        raise HTTPException(status_code=503, detail="RAG pipeline not ready")
    
    try:
        result = chain.invoke(request.question)
        return {
            "question": request.question,
            "answer": result,
            "status": "success"
        }
    except Exception as e:
        print(f"❌ Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint"""
    return {"message": "API working!", "timestamp": "2026-01-31"}

# Render startup confirmation
@app.on_event("startup")
async def startup_event():
    print("🎉 Verilia RAG API startup complete!")
    print(f"✅ Listening on 0.0.0.0:{PORT}")
    print("✅ POST to /query with JSON: {\"question\": \"your question\"}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=port,
        log_level="info",
        workers=1
    )
