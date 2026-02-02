import chromadb

# Initialize the persistent client, specifying the same path used when creating collections
# If using in-memory or a server, initialize the appropriate client (e.g., chromadb.HttpClient)
client = chromadb.PersistentClient(path="./chroma_db") 

# Retrieve a list of all collection objects
collections = client.list_collections()

# Extract and print the names
print("Available collections:")
for collection in collections:
    print(f"* {collection.name}")
