from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough  # ✅ FIXED TYPO
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

app = FastAPI(title="Verilia Devotional RAG API")

# API Key check
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY required")

print("✅ Loading embeddings...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("✅ Loading Chroma DB...")
vector_store = Chroma(
    persist_directory="./chroma_db_free",
    embedding_function=embedding_model
)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})

prompt = ChatPromptTemplate.from_template("""
Answer like a pastor using this context:

{context}

Question: {question}
""")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_devotional(request: QueryRequest):
    try:
        result = chain.invoke(request.question)
        return {"question": request.question, "answer": result, "status": "success"}
    except Exception as e:
        return {"question": request.question, "error": str(e), "status": "error"}

@app.get("/")
async def root():
    return {"message": "Verilia RAG API LIVE! POST to /query"}

@app.get("/health")
async def health():
    return {"status": "healthy"}
