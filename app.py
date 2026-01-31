from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

app = FastAPI(title="Verilia Devotional RAG API")

# API Key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment")

# Request model
class QueryRequest(BaseModel):
    question: str

print("Loading Chroma vector store from disk...")

# Embedding function (used only for similarity search, not generation)
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)

# Load prebuilt vector DB (NO Gemini calls here)
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

# Prompt
prompt = ChatPromptTemplate.from_template("""
Use the following pieces of context to answer the question at the end.
Own the content — do not say "according to the text above."
Answer like a pastor speaking clearly and confidently.

If you don't know the answer, just say you don't know.

First return:
1. The title(s) of the devotional(s) where you got the answer
2. Date(s) of release
3. Then return the answer
4. Return some scriptures for reference

Context:
{context}

Question:
{question}
""")

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

# RAG Chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Routes
@app.post("/query")
async def query_devotional(request: QueryRequest):
    result = chain.invoke(request.question)
    return {
        "question": request.question,
        "answer": result,
        "status": "success"
    }

@app.get("/")
async def root():
    return {"message": "Verilia Devotional RAG API LIVE! POST to /query"}

@app.get("/health")
async def health():
    return {"status": "healthy", "vector_db": "loaded"}
