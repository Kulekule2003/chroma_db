from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()
app = FastAPI(title="Verilia Devotional RAG API")

# API Key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
api_key = os.getenv("GOOGLE_API_KEY")

class QueryRequest(BaseModel):
    question: str

# Global chain setup (runs once on startup)
print("Loading devo.csv and building vector store...")
loader = CSVLoader("devo.csv")
data = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(data)

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=api_key
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# YOUR EXACT PROMPT TEMPLATE
prompt = ChatPromptTemplate.from_template("""
Use the following pieces of context to answer the question at the end. own the content above don't use phrases like according to the text above, act like a pastor that answers questions
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

# YOUR EXACT CHAIN
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

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
    return {"status": "healthy", "data": "devo.csv loaded"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
