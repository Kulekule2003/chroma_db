import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

cols = client.list_collections()
print("Collections:", cols)

if cols:
    col = cols[0]
    print("Using:", col.name)
    print("Document count:", col.count())
