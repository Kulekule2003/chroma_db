# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from itertools import cycle

from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever

# --------------------------
# CONFIG
# --------------------------
DB_DIR = "chroma_db"

# Load your Google API keys as environment variable (comma separated)
API_KEYS = os.environ.get("GOOGLE_API_KEYS", "").split(",")
if not API_KEYS or API_KEYS == [""]:
    raise ValueError("Set GOOGLE_API_KEYS in environment variables (comma separated)")

key_cycle = cycle(API_KEYS)

def get_embedder():
    """Return a Google embedding object (only first key is enough for querying)."""
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=next(key_cycle)
    )

GOOGLE_API_KEY = API_KEYS[0]  # use first key for LLM

# --------------------------
# LOAD VECTORSTORE
# --------------------------
vectorstore = Chroma(
    persist_directory=DB_DIR,
    # embedding_function removed since DB already exists and is only for querying
)

retriever: BaseRetriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

# --------------------------
# PROMPT TEMPLATE
# --------------------------
prompt = ChatPromptTemplate.from_template("""
Use the following pieces of context to answer the question at the end. Own the content above, act like a pastor.
If you don't know the answer, just say that you don't know.
Return:
1. The title(s) of the devotional(s)
2. Date(s) of release
3. Answer to the question
4. Some scriptures for reference
Context: {context}
Question: {question}
""")

# --------------------------
# LLM
# --------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --------------------------
# API
# --------------------------
class Question(BaseModel):
    question: str

app = FastAPI()

@app.post("/chat")
async def chat(q: Question):
    try:
        result = chain.invoke(q.question)
        return {"answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------
# Root endpoint
# --------------------------
@app.get("/")
async def root():
    return {"message": "RAG Chat API is running!"}