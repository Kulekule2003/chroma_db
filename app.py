from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever

# --------------------------
# CONFIG
# --------------------------
DB_DIR = "chroma_db"
API_KEY = os.environ.get("GOOGLE_API_KEY")  # safer for Render

# --------------------------
# LOAD VECTORSTORE (prebuilt embeddings)
# --------------------------
vectorstore = Chroma(
    persist_directory=DB_DIR,
    embedding_function=None  # use existing vectors
)

retriever: BaseRetriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

# --------------------------
# PROMPT TEMPLATE
# --------------------------
prompt = ChatPromptTemplate.from_template("""
Use the following pieces of context to answer the question at the end. own the content above, act like a pastor.
If you don't know the answer, just say that you don't know.
first return
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
    google_api_key=API_KEY
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
