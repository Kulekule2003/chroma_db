import csv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document  # FIXED import

print("🚀 Building FREE Chroma DB...")

# FREE embeddings (works with your versions)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Load CSV - FIRST 50 texts
csv_file = "devo.csv"
print("📥 Loading devo.csv...")

with open(csv_file, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    rows = [row for row in reader]

texts = [row[0] for row in rows[:50] if len(row) > 0]
print(f"✅ Loaded {len(texts)} texts")

# FIXED splitter import + Document creation
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Create documents manually (compatible with 0.2.17)
chunks = []
for i, text in enumerate(texts):
    chunks.append(Document(page_content=text, metadata={"source": f"devo_{i}"}))
print(f"✅ Created {len(chunks)} chunks")

# Build Chroma DB
print("🔨 Building Chroma DB...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db_free"
)

print("✅ SUCCESS! ./chroma_db_free ready for deployment!")
