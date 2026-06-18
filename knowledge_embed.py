"""
knowledge_embed.py

This script reads a dataset from a TSV file, cleans it, processes it into a combined text format, 
and ingests the text embeddings into a TiDB Vector Store using LangChain and HuggingFace.
"""

import os
import pandas as pd
from dotenv import load_dotenv

# LangChain Libraries
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import TiDBVectorStore

# Load environment variables securely from the .env file
load_dotenv()

# ==========================================
# 1. Initialize the Embedder via LangChain
# ==========================================
print("Loading Embedding Model...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# ==========================================
# 2. Setup TiDB Connection String
# ==========================================
print("Preparing TiDB Connection...")
tidb_connection_string = (
    f"mysql+pymysql://{os.getenv('TIDB_USER')}:{os.getenv('TIDB_PASSWORD')}"
    f"@{os.getenv('TIDB_HOST')}:{os.getenv('TIDB_PORT')}/{os.getenv('TIDB_DATABASE')}"
    "?ssl_verify_cert=false&ssl_verify_identity=false"
)

# ==========================================
# 3. Read and Clean the Dataset
# ==========================================
print("Reading dataset...")

# Load the data, skipping any corrupted lines
df = pd.read_csv("data_knowledge.csv", sep="\t", on_bad_lines="skip")

# Clean column headers by stripping trailing/leading whitespaces
df.columns = df.columns.str.strip()

# Drop rows where 'Question' or 'Answer' is missing
df = df.dropna(subset=['Question', 'Answer'])

# Remove duplicate questions to keep the dataset clean and unique
df = df.drop_duplicates(subset=['Question'])

# Combine the columns into a rich text format to provide better context for the AI
df['qa_combined'] = (
    "Topic: " + df['ArticleTitle'].astype(str) + "\n" +
    "Question: " + df['Question'].astype(str) + "\n" +
    "Fact/Answer: " + df['Answer'].astype(str)
)

# Convert the combined text into a Python list
texts_to_embed = df['qa_combined'].tolist()

print(f"Successfully loaded data! {len(texts_to_embed)} rows ready for processing.")

# ==========================================
# 4. Ingest to TiDB Vector Store
# ==========================================
print("Connecting to TiDB and setting up the table...")

# Step A: Open the connection and configure the vector store
vector_store = TiDBVectorStore(
    connection_string=tidb_connection_string,
    table_name="documents",
    distance_strategy="cosine",
    embedding_function=embeddings
)

# Step B: Insert the new texts and their embeddings into the database
print("Ingesting new data into TiDB Cloud... This process may take a few minutes.")
vector_store.add_texts(texts=texts_to_embed)

print(f"Success! All {len(texts_to_embed)} records have been seamlessly added to the database using the LangChain schema.")
