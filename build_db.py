import csv
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

print("🚀 Building FREE Chroma DB for production...")

# FREE LOCAL EMBEDDINGS (no quota limits)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Load CSV (FIRST 50 for quota safety, change to rows[:] for all)
csv_file = "devo.csv"
print("📥 Loading devo.csv...")

with open(csv_file, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    rows = [row for row in reader]

# FIRST 50 ROWS (safe amount)
texts = [row[0] for row in rows[:50] if len(row) > 0]
print(f"✅ Loaded {len(texts)} texts")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.create_documents(texts)
print(f"✅ Created {len(chunks)} chunks")

# Build NEW Chroma DB with FREE embeddings
print("🔨 Building Chroma DB with FREE embeddings...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db_free"  # NEW folder
)

vectorstore.persist()
print("✅ SUCCESS! FREE Chroma DB built in ./chroma_db_free")
print("📁 Copy this folder to your deployment repo!")
