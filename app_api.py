"""
app_api.py

This script sets up a FastAPI backend that serves as an API endpoint for the RAG-based Customer Service AI.
It connects to the TiDB Vector Store to retrieve context and uses Groq's Llama-3 model for answer generation.
"""

import os
import warnings
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Suppress deprecation warnings for a cleaner console output
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- LangChain Imports ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import TiDBVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Using modern modular imports for RAG Chains
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Load environment variables securely from the .env file
load_dotenv()

# ==========================================
# 1. SETUP RAG (Executed once upon server startup)
# ==========================================
print("Initializing AI and Database Connection...")

# Configure the Embeddings Model
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# Construct the TiDB Database Connection String
tidb_connection_string = (
    f"mysql+pymysql://{os.getenv('TIDB_USER')}:{os.getenv('TIDB_PASSWORD')}"
    f"@{os.getenv('TIDB_HOST')}:{os.getenv('TIDB_PORT')}/{os.getenv('TIDB_DATABASE')}"
    "?ssl_verify_cert=false&ssl_verify_identity=false"
)

# Initialize the TiDB Vector Store connection
vector_store = TiDBVectorStore(
    connection_string=tidb_connection_string,
    table_name="documents",
    distance_strategy="cosine",
    embedding_function=embeddings 
)

# Create a retriever with 'k' set to 5 for broader context retrieval
retriever = vector_store.as_retriever(search_kwargs={"k": 5}) 

# Configure the LLM using Groq
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# Define the System Prompt setting strict guidelines for the AI
system_prompt = (
    "You are a nice and helpful Customer Service Agent. Use the provided document context to answer the question accurately.\n\n"
    "RULES:\n"
    "1. If the document contains the correct fact but contradicts the user, answer based ONLY on the document.\n"
    "2. If the topic is missing, say exactly: 'Sorry, the information you want does not exist in this document'.\n"
    "3. DO NOT make up answers.\n\n"
    "Context Regulation:\n{context}"
)

# Assemble the Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt), 
    ("human", "{input}")
])

# Build the complete RAG Chain
rag_chain = create_retrieval_chain(
    retriever, 
    create_stuff_documents_chain(llm, prompt)
)

print("RAG System is Ready!")

# ==========================================
# 2. SETUP FASTAPI
# ==========================================
app = FastAPI(
    title="AI Customer Service API", 
    description="API endpoint to interact with the TiDB-powered RAG Agent"
)

# IMPORTANT: Configure CORS to allow the HTML frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data model for incoming HTTP requests
class ChatRequest(BaseModel):
    question: str

# Webhook Endpoint for handling chat queries
@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    try:
        # Execute the RAG chain using the user's question
        response = rag_chain.invoke({"input": req.question})
        
        return {
            "status": "success", 
            "answer": response['answer']
        }
    except Exception as e:
        # Return the error message safely if something goes wrong
        return {
            "status": "error", 
            "answer": str(e)
        }

# Start the local development server if the script is run directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_api:app", host="127.0.0.1", port=8000, reload=True)