__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import json
from itertools import cycle

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ======================
# CONFIG
# ======================
CSV_FILE = "devo.csv"
DB_DIR = "./chroma_db"
COLLECTION_NAME = "devo_collection"
PROGRESS_FILE = "progress.json"
BATCH_SIZE = 50 

# Replace with your actual keys locally
API_KEYS = [
   "AIzaSyAoCHHJgIbjJwrTAtyLnm2hQJIBd5XP4Ig",
    "AIzaSyBu_TjtJMx1Y5BIyUtqb_kD8Ur99tT5VNo",
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
# LOAD CSV SAFELY
# ======================
print("Loading CSV...")
documents = []

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"Could not find {CSV_FILE}")

with open(CSV_FILE, "rb") as f:
    raw = f.read()

try:
    text = raw.decode("utf-8")
except UnicodeDecodeError:
    text = raw.decode("latin-1")

lines = text.splitlines()
for idx, line in enumerate(lines):
    row_text = line.strip()
    if len(row_text) < 20:
        continue
    doc = Document(page_content=row_text, metadata={"row": idx})
    documents.append(doc)

print("Total rows loaded:", len(documents))

# ======================
# CHUNKING
# ======================
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
chunks = splitter.split_documents(documents)

for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = f"chunk_{i}"

# ======================
# CREATE DB
# ======================
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=DB_DIR,
    embedding_function=get_embedder()
)

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
        print("Error:", e)
        break

print(f"Finished! Database saved to {DB_DIR}")