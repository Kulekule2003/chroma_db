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
DB_DIR = "chroma_db"
PROGRESS_FILE = "progress.json"
BATCH_SIZE = 50  # larger batch for efficiency

API_KEYS = [ ] #comma seperated API Keys for only local experimentation

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
print("Loading CSV with safe encoding...")

documents = []

with open(CSV_FILE, "rb") as f:
    raw = f.read()

try:
    text = raw.decode("utf-8")
except UnicodeDecodeError:
    text = raw.decode("latin-1")

lines = text.splitlines()

for idx, line in enumerate(lines):
    row_text = line.strip()
    if len(row_text) < 20:  # skip tiny or empty rows
        continue

    doc = Document(
        page_content=row_text,
        metadata={"row": idx}
    )
    documents.append(doc)

print("Total rows loaded:", len(documents))

# ======================
# CHUNKING (OPTIMIZED)
# ======================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # bigger chunks for fewer total embeddings
    chunk_overlap=20  # smaller overlap
)

chunks = splitter.split_documents(documents)
print("Total chunks after optimization:", len(chunks))

# ======================
# ASSIGN STABLE IDS
# ======================
for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = f"chunk_{i}"

# ======================
# LOAD OR CREATE DB
# ======================
vectorstore = Chroma(
    persist_directory=DB_DIR,
    embedding_function=get_embedder()
)

# ======================
# HELPER: SAVE PROGRESS
# ======================
def save_progress():
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(list(completed_ids), f)

# ======================
# EMBEDDING LOOP
# ======================
for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i + BATCH_SIZE]

    to_add = []
    for c in batch:
        cid = c.metadata["chunk_id"]
        if cid not in completed_ids:
            to_add.append(c)

    if not to_add:
        continue

    try:
        print(f"Embedding {len(to_add)} chunks...")
        vectorstore.add_documents(to_add)  # Chroma auto-saves

        for c in to_add:
            completed_ids.add(c.metadata["chunk_id"])

        save_progress()

    except Exception as e:
        print("API/Quota error:", e)
        print("You can safely rerun the script to resume.")
        break

print("Embedding process finished")
print("Total embedded chunks:", len(completed_ids))
