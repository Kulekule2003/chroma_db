from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import sqlite3

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your React app URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for RAG pipeline
vector_store = None
chain = None
api_key = os.getenv("GOOGLE_API_KEY")

class Query(BaseModel):
    question: str

def init_rag_pipeline():
    global vector_store, chain
    
    # Load your CSV (upload to repo or use Render Disks for persistence)
    loader = CSVLoader("devo.csv")
    data = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(data)
    
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", google_api_key=api_key
    )
    
    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding_model, persist_directory="./chroma_db"
    )
    
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    
    prompt = ChatPromptTemplate.from_template("""
    Use the following pieces of context to answer the question at the end. Own the content above don't use phrases like according to the text above, act like a pastor that answers questions
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    first return
    1. The title(s) of the devotional(s) where you getting the answer
    2. Date(s) of release
    3. Then return the answer to the question
    4. return some scriptures for reference
    Context: {context}
    Question: {question}
    """)
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

@app.on_event("startup")
async def startup_event():
    init_rag_pipeline()

@app.post("/query")
async def query_rag(query: Query):
    if chain is None:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized")
    try:
        result = chain.invoke(query.question)
        return {"answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
