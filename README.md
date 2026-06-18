# 🚀 TiDB-Powered RAG Customer Service Agent

<div align="center">
  <img src="https://img.shields.io/badge/Language-Python-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Framework-FastAPI-009688.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/Database-TiDB_Cloud-blueviolet.svg" alt="TiDB">
  <img src="https://img.shields.io/badge/LLM-Groq-red.svg" alt="Groq">
</div>

---

## 📖 Project Overview
This project is an end-to-end **Retrieval-Augmented Generation (RAG)** system built to act as a smart Customer Service Agent. It features a modern minimalist web interface, a robust FastAPI backend, LangChain orchestration, and persistent vector storage using TiDB Cloud.

---

## ⚙️ 1. Set Up Groq API Key
To power the AI with the blazing-fast Llama-3 model, you need a Groq API key:
1. Go to the [Groq Cloud Console](https://console.groq.com/).
2. Create an account or log in.
3. Navigate to the **API Keys** section and generate a new key. Keep it secret!

## 🗄️ 2. Set Up TiDB Vector Database
We use TiDB Cloud to store our knowledge base and perform semantic vector searches:
1. Sign up for a free cluster at [TiDB Cloud](https://tidbcloud.com/).
2. Create a new Serverless cluster.
3. Once created, click **Connect** and gather your host, port, username, and password.

## 🔐 3. Environment Variables (`.env`)
Create a hidden file named `.env` in the root directory of this project. Do **NOT** upload this file to GitHub. Add your credentials in the following format:

```env
GROQ_API_KEY=your_groq_api_key_here
TIDB_HOST=your_tidb_cluster_host.tidbcloud.com
TIDB_PORT=4000
TIDB_USER=your_tidb_username
TIDB_PASSWORD=your_tidb_password
TIDB_DATABASE=your_database_name
```

## 📦 4. Install Dependencies
Before running the scripts, install all the required Python libraries. Open your terminal and run:
```Bash
pip install pandas fastapi uvicorn pydantic python-dotenv langchain langchain-core langchain-huggingface langchain-groq langchain-community pymysql
```
## 🧠 5. Understanding knowledge_embed.py
This script acts as the Data Ingestion Pipeline.
It reads your raw dataset (data_knowledge.csv), cleans formatting errors, and removes duplicate questions using Pandas. It then merges the topic, question, and answer into a cohesive text block, converts them into mathematical vectors using the BAAI/bge-m3 embedding model, and seamlessly pushes them into your TiDB Cloud database.

## 🔌 6. Understanding app_api.py
This script is the Brain & Backend of the application.
It spins up a FastAPI server that listens for incoming chat messages from the web frontend. When a question is received, it triggers the LangChain pipeline: it searches the TiDB database for the most relevant facts, combines them with a strict Customer Service System Prompt, and asks Groq's Llama-3 model to generate a highly accurate, hallucination-free response.

## ▶️ 7. How to Run the Code
Follow these steps sequentially to launch the system:

Step A: Populate the Database
Run the ingestion script first. You only need to do this once (or whenever your CSV data changes).

```Bash
python knowledge_embed.py
```
(Wait until the terminal shows a "Success" message indicating data is saved to TiDB).

Step B: Start the Backend Server
Once the database is ready, start the FastAPI server:

```Bash
python app_api.py
```
(Keep this terminal open and running. You should see Uvicorn running on http://127.0.0.1:8000).

🌐 8. Test the Results (index.html)
With the backend running, simply double-click the index.html file to open it in your favorite web browser (Chrome, Safari, Edge, etc.).
Type a question based on your dataset into the chat box, hit enter, and watch your TiDB-powered AI respond in real-time through the clean, minimalist UI!
