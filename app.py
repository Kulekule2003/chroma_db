from fastapi import FastAPI
from pydantic import BaseModel
import os
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

app = FastAPI(title="Verilia Devotional RAG API")

# API Key (ONLY for LLM generation)
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment")

print("Loading FREE HuggingFace embeddings...")
# ✅ FREE EMBEDDINGS - NO GEMINI QUOTA
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Loading Chroma vector store...")
# ✅ Load with matching FREE embeddings
vector_store = Chroma(
    persist_directory="./chroma_db_free",
    embedding_function=embedding_model
)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})

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

# ✅ Gemini ONLY for text generation (has quota)
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
    return {"message": "Verilia Devotional RAG API LIVE! POST to /query"}

@app.get("/health")
async def health():
    return {"status": "healthy", "vector_db": "loaded"}
