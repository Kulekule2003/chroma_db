import os
import csv
import json
from itertools import cycle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from langchain_chroma import Chroma
from chromadb import Client
from chromadb.config import Settings

# ──────────────────────────
# ENV / PATHS
# ──────────────────────────
os.environ["ANONYMIZED_TELEMETRY"] = "False"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join("/tmp", "chroma_db")  # Render writable dir
CSV_PATH = os.path.join(BASE_DIR, "devo.csv")

# ──────────────────────────
# FASTAPI APP
# ──────────────────────────
app = FastAPI(title="AI Scriptural Counsellor")

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
# Last key reserved for user queries
# Set in Render: GOOGLE_API_KEYS="BUILD1,BUILD2,...,USER_KEY"
# ──────────────────────────
RAW_KEYS = os.environ.get("GOOGLE_API_KEYS", "")
API_KEYS = [k.strip() for k in RAW_KEYS.split(",") if k.strip()]
if len(API_KEYS) < 1:
    raise RuntimeError("GOOGLE_API_KEYS must contain at least one key")

BUILD_KEYS = API_KEYS[:-1] if len(API_KEYS) > 1 else API_KEYS
USER_KEY = API_KEYS[-1]
build_key_cycle = cycle(BUILD_KEYS)

def get_embedder():
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=next(build_key_cycle)
    )

# ──────────────────────────
# HELPERS
# ──────────────────────────
def format_docs(docs):
    if not docs:
        return "No devotional context found in database."
    return "\n\n".join(
        f"Title: {doc.metadata.get('title','Unknown')}\n"
        f"Date: {doc.metadata.get('date','Unknown')}\n"
        f"Content:\n{doc.page_content}"
        for doc in docs
    )

def rebuild_db_safe():
    """Build Chroma DB safely with rotating API keys"""
    print("⚠️ Chroma DB missing — attempting rebuild")

    if not os.path.exists(CSV_PATH):
        print("❌ devo.csv not found — skipping rebuild")
        return False

    try:
        # Load CSV
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [h.strip() for h in reader.fieldnames]
            documents = []
            for idx, row in enumerate(reader):
                content = row.get("Body", "").strip()
                if len(content) < 20:
                    continue
                doc = Document(
                    page_content=content,
                    metadata={
                        "title": row.get("Title","").strip(),
                        "date": row.get("Date","").strip(),
                        "theme": row.get("Theme","").strip(),
                        "scripture": row.get("Theme Scripture","").strip(),
                        "row": idx
                    }
                )
                documents.append(doc)

        if not documents:
            print("❌ No documents loaded from CSV")
            return False

        # Chunking
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        chunks = splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"chunk_{i}"

        print(f"Loaded {len(documents)} docs, {len(chunks)} chunks")

        # Chroma client & collection
        chroma_client = Client(Settings(persist_directory=DB_DIR, anonymized_telemetry=False))
        try:
            collection = chroma_client.get_collection("devotionals")
        except:
            collection = chroma_client.create_collection("devotionals")

        vectorstore = Chroma(
            client=chroma_client,
            collection_name="devotionals",
            embedding_function=get_embedder()
        )

        # Embedding loop
        BATCH_SIZE = 50
        completed_ids = set()
        PROGRESS_FILE = os.path.join(BASE_DIR, "progress.json")
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                completed_ids = set(json.load(f))

        def save_progress():
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump(list(completed_ids), f)

        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i+BATCH_SIZE]
            to_add = [c for c in batch if c.metadata["chunk_id"] not in completed_ids]
            if not to_add:
                continue
            try:
                print(f"Embedding batch {i}-{i+len(to_add)}")
                vectorstore.add_documents(to_add)
                for c in to_add:
                    completed_ids.add(c.metadata["chunk_id"])
                save_progress()
            except Exception as e:
                print("⚠️ Embedding batch failed (quota issue?):", e)
                break

        print(f"✅ DB built safely with {collection.count()} documents")
        return True

    except Exception as e:
        print("❌ Failed to rebuild DB:", e)
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

    # Build DB if missing or empty
    if not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
        rebuild_db_safe()

    # Chroma client
    chroma_client = Client(Settings(persist_directory=DB_DIR, anonymized_telemetry=False))
    collections = chroma_client.list_collections()
    collection_name = collections[0].name if collections else None

    if collection_name:
        vectorstore = Chroma(client=chroma_client, collection_name=collection_name,
                             embedding_function=GoogleGenerativeAIEmbeddings(
                                 model="models/text-embedding-004",
                                 google_api_key=USER_KEY
                             ))
        doc_count = chroma_client.get_collection(collection_name).count()
    else:
        vectorstore = None
        doc_count = 0

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2}) if vectorstore and doc_count>0 else None

    # LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=USER_KEY)
    prompt = ChatPromptTemplate.from_template("""
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
""")

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
# API MODELS & ROUTES
# ──────────────────────────
class Question(BaseModel):
    question: str

@app.get("/")
async def root():
    return {"status": "RAG Chat API running", "user_key_reserved": True}

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
                {"name": c.name, "documents": chroma_client.get_collection(c.name).count()}
                for c in collections
            ],
        }
    except Exception as e:
        return {"error": str(e)}
