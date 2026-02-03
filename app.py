# app.py
"""FastAPI entrypoint for Render deployment."""

# Ensure Chroma uses bundled SQLite on hosts like Render
__import__("pysqlite3")  # type: ignore[import-not-found]
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import os
import csv
import json
import warnings
import shutil
from itertools import cycle
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from chromadb import Client
from chromadb.config import Settings

# ──────────────────────────
# CONFIGURATION
# ──────────────────────────
warnings.filterwarnings("ignore", message=".*telemetry.*")
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# ──────────────────────────
# PATHS
# ──────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join("/tmp", "chroma_db")  # Render ephemeral dir
CSV_PATH = os.path.join(BASE_DIR, "devo.csv")
PROGRESS_FILE = os.path.join(BASE_DIR, "progress.json")

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
# Comma separated: KEY1,KEY2,...,LAST_KEY (user)
# ──────────────────────────
RAW_KEYS = os.environ.get("GOOGLE_API_KEYS", "")
API_KEYS = [k.strip() for k in RAW_KEYS.split(",") if k.strip()]

if len(API_KEYS) < 1:
    raise RuntimeError(
        "Environment variable GOOGLE_API_KEYS must contain at least one key "
        "(comma separated; last key is reserved for user traffic)."
    )

BUILD_KEYS = API_KEYS[:-1] if len(API_KEYS) > 1 else API_KEYS
USER_KEY = API_KEYS[-1]

build_key_cycle = cycle(BUILD_KEYS or [USER_KEY])

# Globals initialised safely so routes don't crash before startup
chroma_client = None
vectorstore = None
retriever: BaseRetriever | None = None  # type: ignore[valid-type]
chain = None

def get_embedder():
    """Get embedding function with next build key"""
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=next(build_key_cycle),
    )

# ──────────────────────────
# HELPERS
# ──────────────────────────
def format_docs(docs):
    """Format retrieved documents for context"""
    if not docs:
        return "No devotional context found in database."
    return "\n\n".join(
        f"Title: {doc.metadata.get('title','Unknown')}\n"
        f"Date: {doc.metadata.get('date','Unknown')}\n"
        f"Theme: {doc.metadata.get('theme','Unknown')}\n"
        f"Scripture: {doc.metadata.get('scripture','Unknown')}\n"
        f"Further Study: {doc.metadata.get('further_study','')}\n"
        f"Content:\n{doc.page_content}"
        for doc in docs
    )

def rebuild_db_safe():
    """Build Chroma DB safely on Render, rotating build keys"""
    print("⚠️ Chroma DB missing — attempting rebuild")

    if not os.path.exists(CSV_PATH):
        print("❌ devo.csv not found — skipping rebuild")
        return False

    try:
        # Clean existing DB directory to avoid conflicts
        if os.path.exists(DB_DIR):
            print(f"Cleaning existing DB directory: {DB_DIR}")
            shutil.rmtree(DB_DIR)
        os.makedirs(DB_DIR, exist_ok=True)

        # Load CSV documents
        documents = []
        with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [h.strip() for h in reader.fieldnames]
            for idx, row in enumerate(reader):
                content = row.get("Body", "").strip()
                if len(content) < 20:
                    continue
                doc = Document(
                    page_content=content,
                    metadata={
                        "title": row.get("Title", "").strip(),
                        "date": row.get("Date", "").strip(),
                        "theme": row.get("Theme", "").strip(),
                        "scripture": row.get("Theme Scripture", "").strip(),
                        "further_study": row.get("Further Study", "").strip(),
                        "row": idx,
                    },
                )
                documents.append(doc)

        if not documents:
            print("❌ No documents loaded from CSV")
            return False

        # Chunk documents
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        chunks = splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"chunk_{i}"

        print(f"✅ Loaded {len(documents)} docs, {len(chunks)} chunks")

        # Try each build key until one works
        vectorstore = None
        success = False
        
        for build_key in BUILD_KEYS:
            try:
                key_suffix = build_key[-8:] if len(build_key) > 8 else build_key
                print(f"🔄 Attempting to create vectorstore with key ending in ...{key_suffix}")
                
                # Create embeddings function with current key
                embedder = GoogleGenerativeAIEmbeddings(
                    model="models/text-embedding-004",
                    google_api_key=build_key,
                )
                
                # Create vectorstore directly (Chroma handles persistence)
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embedder,
                    persist_directory=DB_DIR,
                    collection_name="devotionals",
                )

                # Explicitly persist when supported (older langchain-chroma)
                if hasattr(vectorstore, "persist"):
                    vectorstore.persist()
                else:
                    print("ℹ️ Chroma instance has no 'persist' method; assuming auto-persistence.")
                
                success = True
                print(f"✅ DB created successfully with key ending in ...{key_suffix}")
                break
                
            except Exception as e:
                error_msg = str(e)
                if "quota" in error_msg.lower() or "limit" in error_msg.lower():
                    print(f"⛔ Key ...{key_suffix} quota exceeded, trying next")
                else:
                    print(f"⚠️ Key ...{key_suffix} failed: {error_msg[:100]}")
                continue
        
        if not success:
            print("❌ All build keys failed")
            return False
        
        # Verify the collection was created
        try:
            client = Client(Settings(persist_directory=DB_DIR))
            collection = client.get_collection("devotionals")
            count = collection.count()
            print(f"✅ DB verification: {count} documents in collection")
            
            # Save progress for future reference
            progress_data = {
                "total_documents": len(documents),
                "total_chunks": len(chunks),
                "collection_count": count,
                "rebuild_time": json.dumps(str(os.path.getmtime(CSV_PATH)))
            }
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump(progress_data, f)
                
            return count > 0
            
        except Exception as e:
            print(f"❌ DB verification failed: {e}")
            return False

    except Exception as e:
        print(f"❌ Failed to rebuild DB: {e}")
        import traceback
        traceback.print_exc()
        return False

# ──────────────────────────
# STARTUP - Modified to be synchronous for Render compatibility
# ──────────────────────────
@app.on_event("startup")
def startup():
    """Initialize RAG system on startup"""
    global chroma_client, vectorstore, retriever, chain

    print("\n" + "="*60)
    print("RAG STARTUP INITIALIZED")
    print("="*60)
    print(f"BASE DIR: {BASE_DIR}")
    print(f"DB PATH: {DB_DIR}")
    print(f"BUILD KEYS: {len(BUILD_KEYS)}")
    print(f"USER KEY RESERVED: True")
    
    # Check if DB needs rebuilding
    needs_rebuild = False
    chroma_client = None
    
    try:
        chroma_client = Client(Settings(persist_directory=DB_DIR, anonymized_telemetry=False))
        collections = chroma_client.list_collections()
        
        if not collections:
            print("⚠️ No collections found in DB")
            needs_rebuild = True
        else:
            collection_name = collections[0].name
            doc_count = chroma_client.get_collection(collection_name).count()
            print(f"📊 Found collection '{collection_name}' with {doc_count} documents")
            
            if doc_count == 0:
                print("⚠️ Collection exists but is empty")
                needs_rebuild = True
            elif doc_count < 100:  # Arbitrary threshold
                print(f"⚠️ Low document count ({doc_count}), checking CSV...")
                if os.path.exists(CSV_PATH):
                    # Quick check if CSV has more content
                    with open(CSV_PATH, 'r', encoding='utf-8-sig') as f:
                        reader = csv.DictReader(f)
                        csv_rows = sum(1 for _ in reader)
                        if csv_rows > doc_count:
                            print(f"⚠️ CSV has {csv_rows} rows but DB has {doc_count} docs")
                            needs_rebuild = True
                
    except Exception as e:
        print(f"⚠️ Could not connect to DB: {e}")
        needs_rebuild = True
    
    # Rebuild if needed
    if needs_rebuild:
        print("🔨 Initiating database rebuild...")
        success = rebuild_db_safe()
        if not success:
            print("❌ Database rebuild failed - starting with limited functionality")
    else:
        print("✅ Database appears healthy")
    
    # Re-initialize client after possible rebuild
    try:
        chroma_client = Client(Settings(persist_directory=DB_DIR, anonymized_telemetry=False))
        collections = chroma_client.list_collections()
        
        if not collections:
            print("⚠️ Running in EMPTY MODE - no vectorstore available")
            vectorstore = None
            retriever = None
            doc_count = 0
        else:
            collection_name = collections[0].name
            doc_count = chroma_client.get_collection(collection_name).count()
            
            # Initialize vectorstore with USER key for queries
            embedder = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=USER_KEY,
            )
            
            vectorstore = Chroma(
                client=chroma_client,
                collection_name=collection_name,
                embedding_function=embedder,
                persist_directory=DB_DIR,
            )
            
            print(f"✅ Vectorstore initialized with {doc_count} documents")
            
    except Exception as e:
        print(f"❌ Failed to initialize vectorstore: {e}")
        vectorstore = None
        retriever = None
        doc_count = 0
    
    # Retriever setup
    if vectorstore and doc_count > 0:
        retriever = vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 3}
        )
        print(f"✅ Retriever ready (k=3)")
    else:
        retriever = None
        print("⚠️ No retriever available - using fallback responses")
    
    # LLM setup with USER key
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=USER_KEY,
        temperature=0.3
    )
    
    # Prompt template
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
    
    # Safe retrieval function
    def safe_retrieve(question):
        if not retriever:
            return []
        try:
            results = retriever.invoke(question)
            print(f"🔍 Retrieved {len(results)} documents for query: '{question[:50]}...'")
            return results
        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            return []
    
    # Chain construction
    chain = (
        {
            "context": RunnableLambda(safe_retrieve) | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("="*60)
    print("✅ RAG SYSTEM READY")
    print("="*60 + "\n")
    
    # IMPORTANT: Add this to ensure the server starts properly
    print("🚀 FastAPI application fully initialized and ready to accept requests")

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
    """Root endpoint"""
    try:
        doc_count = 0
        if chroma_client:
            collections = chroma_client.list_collections()
            if collections:
                doc_count = chroma_client.get_collection(collections[0].name).count()
        
        return {
            "status": "RAG Chat API running",
            "version": "1.0.0",
            "database": {
                "documents": doc_count,
                "path": DB_DIR,
                "exists": os.path.exists(DB_DIR)
            },
            "keys": {
                "build_keys": len(BUILD_KEYS),
                "user_key_reserved": True
            }
        }
    except Exception as e:
        return {"status": "running", "error": str(e)}


@app.head("/")
async def root_head():
    """HEAD handler for platform health checks (e.g. Render)."""
    return Response(status_code=200)

@app.post("/chat")
async def chat(q: Question):
    """Main chat endpoint"""
    try:
        if chain is None:
            raise HTTPException(
                status_code=503,
                detail="RAG system is not initialized yet. Please try again shortly.",
            )
        print(f"💬 Chat request: '{q.question[:100]}...'")
        result = chain.invoke(q.question)
        return {"answer": result}
    except Exception as e:
        print(f"[CHAT ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug-db")
async def debug_db():
    """Debug database endpoint"""
    try:
        collections = chroma_client.list_collections()
        return {
            "db_path": DB_DIR,
            "exists": os.path.exists(DB_DIR),
            "collections": [
                {
                    "name": c.name, 
                    "documents": chroma_client.get_collection(c.name).count(),
                    "metadata": chroma_client.get_collection(c.name).metadata
                }
                for c in collections
            ],
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        if chroma_client is None:
            return {
                "status": "unhealthy",
                "reason": "chroma_client not initialized",
            }, 503

        doc_count = 0
        collections = chroma_client.list_collections()
        if collections:
            doc_count = chroma_client.get_collection(collections[0].name).count()
        
        return {
            "status": "healthy" if doc_count > 0 else "degraded",
            "database": {
                "documents": doc_count,
                "collections": len(chroma_client.list_collections()) if chroma_client else 0
            },
            "timestamp": json.dumps(str(os.path.getmtime(__file__) if os.path.exists(__file__) else 0))
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}, 503

@app.get("/debug-collection")
async def debug_collection():
    """Detailed collection inspection"""
    try:
        if not chroma_client:
            return {"error": "Chroma client not initialized"}
            
        collections = chroma_client.list_collections()
        result = {
            "db_path": DB_DIR,
            "exists": os.path.exists(DB_DIR),
            "collections_count": len(collections),
            "build_keys_count": len(BUILD_KEYS),
            "csv_exists": os.path.exists(CSV_PATH)
        }
        
        for collection in collections:
            coll = chroma_client.get_collection(collection.name)
            result[collection.name] = {
                "count": coll.count(),
                "metadata": coll.metadata,
                "sample_ids": coll.get(limit=2)["ids"] if coll.count() > 0 else []
            }
        
        # Add CSV info
        if os.path.exists(CSV_PATH):
            with open(CSV_PATH, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                csv_rows = sum(1 for _ in reader)
                result["csv_rows"] = csv_rows
        
        return result
    except Exception as e:
        return {"error": str(e)}

@app.get("/reset-db")
async def reset_db():
    """Force reset database (use with caution)"""
    try:
        if os.path.exists(DB_DIR):
            shutil.rmtree(DB_DIR)
            print(f"🗑️ Removed DB directory: {DB_DIR}")
        
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
            print(f"🗑️ Removed progress file: {PROGRESS_FILE}")
        
        return {
            "status": "Database reset initiated",
            "next_steps": "Restart the application or wait for auto-rebuild"
        }
    except Exception as e:
        return {"error": str(e)}

# ──────────────────────────
# CRITICAL: Main guard for Render
# ──────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"🌐 Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)