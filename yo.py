from langchain_chroma import Chroma
import os

DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

# List all collections in the DB folder
try:
    # Initialize without specifying a collection first
    temp_store = Chroma(persist_directory=DB_DIR)
    print("Collections in DB:", temp_store.list_collections())
except Exception as e:
    print("Error reading DB:", e)
