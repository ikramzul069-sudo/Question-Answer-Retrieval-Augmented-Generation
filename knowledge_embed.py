import os
import json
import pandas as pd
import mysql.connector
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables securely
load_dotenv()

# 1. Initialize the Embedder
print("Loading Embedding Model...")
embedder = SentenceTransformer('BAAI/bge-m3')

# 2. Securely connect to TiDB Cloud 
print("Connecting to TiDB Cloud...")
#You can generate this connector via TiDB
db = mysql.connector.connect(
    host=os.getenv("TIDB_HOST"),
    port=os.getenv("TIDB_PORT"),
    user=os.getenv("TIDB_USER"),
    password=os.getenv("TIDB_PASSWORD"),
    database=os.getenv("TIDB_DATABASE"),
    ssl_ca="<CA_PATH>", # Update this if a specific CA certificate is required
    ssl_verify_cert=False,
    ssl_verify_identity=False
)

curr = db.cursor()

# 3. Read and Clean the Dataset
print("Reading dataset...")
# Using '\t' to prevent splitting spaces within sentences. 
# 'on_bad_lines="skip"' automatically drops corrupted rows.
df = pd.read_csv("data_knowledge.csv", sep="\t", on_bad_lines="skip")

# Clean column names from hidden whitespaces
df.columns = df.columns.str.strip()
df = df.dropna(subset=['Question', 'Answer'])

# --- EMERGENCY DIAGNOSTICS ---
print("\n--- COLUMN DETECTION RESULT ---")
print(df.columns.tolist())
print("---------------------------\n")

# 4. Process and Embed Data
try:
    # Combine Question and Answer columns
    df['qa_combined'] = df['Question'] + "\t" + df['Answer']
    texts_to_embed = df['qa_combined'].tolist()
    
    print(f"Successfully loaded data! {len(texts_to_embed)} rows ready for processing.")
    
    # Batch Embedding Processing
    print("Processing embeddings... This may take a while.")
    embedding_list = embedder.encode(texts_to_embed).tolist()
    
    print("Embedding completed successfully!")
    print(f"Total vectors generated: {len(embedding_list)}")

    # 5. Database Insertion
    sql_query = "INSERT INTO documents (text, embedding) VALUES (%s, %s)"
    print("Saving data to TiDB Cloud...")
    
    # Iterate to insert data securely using parameterized queries
    for text, embedding in zip(texts_to_embed, embedding_list):
        # Convert vector list [0.1, 0.2, ...] into JSON string format
        embedding_str = json.dumps(embedding)
        curr.execute(sql_query, (text, embedding_str))

    # Commit transactions to the database
    db.commit()
    print(f"Success! A total of {len(texts_to_embed)} records have been added to the database.")

except KeyError as e:
    print(f"[FAILED] Column {e} not found.")
    print("Available columns in your current file are:", df.columns.tolist())
except mysql.connector.Error as err:
    print(f"[DATABASE ERROR] A MySQL/TiDB error occurred: {err}")
finally:
    # Safely close database connections
    curr.close()
    db.close()