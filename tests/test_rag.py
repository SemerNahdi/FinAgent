# test_rag.py
from services.rag.rag_tool import RAGTool

# Initialize RAG tool
rag = RAGTool()

# Ingest a small folder with 1-2 files
rag.add_directory("./data")  # point to a test folder

# Run a test query
query = "What was Coca-Cola's Q3 2025 revenue?"
answer = rag.query(query)
print("Query:", query)
print("Answer:", answer)
