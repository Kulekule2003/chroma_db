from chromadb import Client
from chromadb.config import Settings

DB_DIR = "chroma_db"  # your local DB path

client = Client(Settings(persist_directory=DB_DIR, anonymized_telemetry=False))

# List collections
collections = client.list_collections()
print("Collections:", [c.name for c in collections])

# Pick the first collection (or any you want)
collection_name = collections[0].name
collection = client.get_collection(collection_name)

# Inspect documents
docs = collection.get()
for i, doc in enumerate(docs['documents']):
    print(f"Doc {i}:")
    print("  id:", docs['ids'][i])
    print("  content snippet:", docs['documents'][i][:100])
    print("  metadata:", docs['metadatas'][i])
