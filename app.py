from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
app = FastAPI(title="Verilia Devotional RAG API")

# Google Gemini Setup
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-pro")

# ChromaDB Client
client = chromadb.PersistentClient(path="./chroma_db")
try:
    collection = client.get_collection("devotional_docs")
except:
    collection = client.create_collection("devotional_docs")

# YOUR EXACT PROMPT TEMPLATE
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

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_devotional(request: QueryRequest):
    try:
        results = collection.query(
            query_texts=[request.question],
            n_results=3,
            include=["documents", "metadatas"]
        )
        context = "\n".join(results['documents'][0]) if results['documents'] else "No context available"
        metadata = results['metadatas'][0] if results['metadatas'] else []
        
        # Extract titles and dates from metadata
        titles = [m.get('title', 'Unknown') for m in metadata]
        dates = [m.get('date', 'Unknown') for m in metadata]
        
    except:
        context = "No documents loaded yet"
        titles = ["No documents"]
        dates = ["No dates"]
    
    # Format context with titles/dates for prompt
    formatted_context = f"Titles: {', '.join(titles)}\nDates: {', '.join(dates)}\n\n{context}"
    
    chain = prompt | llm
    response = chain.invoke({"context": formatted_context, "question": request.question})
    
    return {
        "question": request.question,
        "answer": response.content,
        "context_used": formatted_context[:400] + "..." if len(formatted_context) > 400 else formatted_context,
        "titles": titles,
        "dates": dates,
        "status": "success"
    }

@app.get("/")
async def root():
    return {"message": "Verilia Devotional RAG API LIVE! POST to /query"}

@app.get("/health")
async def health():
    return {"status": "healthy", "chromadb": "ready"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
