from fastapi import FastAPI
from pydantic import BaseModel
import os
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthroug
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings  # NEW: Fixed deprecation

app = FastAPI(title="Verilia Devotional RAG API")

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found")

print("Loading lightweight HuggingFace embeddings...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading Chroma vector store...")
vector_store = Chroma(
    persist_directory="./chroma_db_free",
    embedding_function=embedding_model
)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})

prompt = ChatPromptTemplate.from_template("""
Use the following context to answer like a pastor.

Context: {context}
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
    return {"message": "Verilia Devotional RAG API LIVE! POST to /query"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
