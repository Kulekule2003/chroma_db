import os
import csv
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # Fixed import
from langchain.schema import Document

# -----------------------------
# Set your API key here
# -----------------------------
GOOGLE_API_KEY = "AIzaSyBjlHLXqKfAdoXZ4asHosu00hiHJkcabS4"

# -----------------------------
# Load CSV - ONLY FIRST 50 TEXTS (quota safe)
# -----------------------------
csv_file = "devo.csv"
print("Loading FIRST 50 rows from devo.csv...")

with open(csv_file, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    rows = [row for row in reader]

# **LIMIT TO 50 ROWS** - safe for free tier quota
texts = [row[0] for row in rows[:50] if len(row) > 0]
print(f"Processing FIRST 50 rows only: {len(texts)} texts")

# -----------------------------
# Split text into chunks
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.create_documents(texts)
print(f"Total chunks from 50 texts: {len(chunks)}")

# -----------------------------
# Initialize embeddings with rate limiting
# -----------------------------
print("Initializing Gemini embeddings...")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    api_key=GOOGLE_API_KEY
)

# -----------------------------
# Build Chroma DB with batching (10 at a time)
# -----------------------------
print("Building Chroma vector store (50 docs with batching)...")

# Create new directory for 50-doc version
persist_dir = "./chroma_db_50"

vectorstore = Chroma(
    persist_directory=persist_dir,
    embedding_function=embeddings
)

# Add in small batches with delays
BATCH_SIZE = 10
for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i+BATCH_SIZE]
    print(f"Embedding batch {i//BATCH_SIZE + 1}/{(len(chunks)-1)//BATCH_SIZE + 1} ({len(batch)} docs)...")
    
    try:
        vectorstore.add_documents(batch)
        print(f"✅ Batch {i//BATCH_SIZE + 1} completed")
        time.sleep(2)  # 2s delay between batches
    except Exception as e:
        print(f"❌ Batch {i//BATCH_SIZE + 1} failed: {str(e)[:100]}")
        time.sleep(60)  # Wait 1 min on error
        continue

print(f"\n✅ SUCCESS! Chroma DB created with {len(chunks)} chunks from FIRST 50 devotionals")
print(f"📁 Stored in: {persist_dir}")
print("🔍 Ready for RAG queries!")
