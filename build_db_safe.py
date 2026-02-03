__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import json
from itertools import cycle
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

import csv

# ======================
# CONFIG
# ======================
CSV_FILE = "devo.csv"                  # your CSV file
DB_DIR = "./chroma_db"                 # directory to store Chroma DB
COLLECTION_NAME = "devo_collection"   # collection name in Chroma
PROGRESS_FILE = "progress.json"        # tracks embedded chunks
BATCH_SIZE = 50                        # number of chunks per batch

# ======================
# GOOGLE API KEYS
# ======================
API_KEYS = [
    "AIzaSyCHvMYBKa59BJ9gHeMub7FRFS0sQMDhoio",
    "AIzaSyBW7iBsamSX0b5DSb18g1ZeZ6tBSOAhku8",
    "AIzaSyBq7FCSJXdD1gc5-my9iebIH0eFDwwyT40",
    "AIzaSyBlz-ExdB6u3GPNYVUgbpkKZx7pSWr3HWk",
    "AIzaSyCkEdJJVJZU1gNX0r8R08iB6MjYzYcc2-w",
    "AIzaSyCel4FzVfiS1DMM-quYz8O8zLQqnnDB84E"
]
key_cycle = cycle(API_KEYS)

def get_embedder():
    """Return Google embedding object with rotating API keys."""
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=next(key_cycle)
    )

# ======================
# LOAD PROGRESS
# ======================
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
        completed_ids = set(json.load(f))
else:
    completed_ids = set()

# ======================
# LOAD CSV
# ======================
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"{CSV_FILE} not found")

print("Loading CSV...")
documents = []

with open(CSV_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    reader.fieldnames = [h.strip() for h in reader.fieldnames]  # remove spaces
    for idx, row in enumerate(reader):
        content = row.get("Body", "").strip()
        if len(content) < 20:
            continue  # skip very short entries
        doc = Document(
            page_content=content,
            metadata={
                "title": row.get("Title", "").strip(),
                "date": row.get("Date", "").strip(),
                "theme": row.get("Theme", "").strip(),
                "scripture": row.get("Theme Scripture", "").strip(),
                "row": idx
            }
        )
        documents.append(doc)

print(f"Total documents loaded: {len(documents)}")

# ======================
# CHUNKING
# ======================
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
chunks = splitter.split_documents(documents)

for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = f"chunk_{i}"

print(f"Total chunks created: {len(chunks)}")

# ======================
# CREATE VECTORSTORE
# ======================
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=DB_DIR,
    embedding_function=get_embedder()
)

# ======================
# HELPER: save progress
# ======================
def save_progress():
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(list(completed_ids), f)

# ======================
# EMBEDDING LOOP
# ======================
for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i + BATCH_SIZE]
    to_add = [c for c in batch if c.metadata["chunk_id"] not in completed_ids]

    if not to_add:
        continue

    try:
        print(f"Embedding {len(to_add)} chunks...")
        vectorstore.add_documents(to_add)
        for c in to_add:
            completed_ids.add(c.metadata["chunk_id"])
        save_progress()
    except Exception as e:
        print("Error during embedding:", e)
        print("Stopping to avoid API quota issues.")
        break

print(f"✅ Finished! Database saved to {DB_DIR}")
