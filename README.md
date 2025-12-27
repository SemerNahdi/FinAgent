# Finance Agent: Multi-Agent Financial Assistant

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Groq](https://img.shields.io/badge/Groq-f3d03e?style=for-the-badge&logo=groq&logoColor=black)](https://groq.com/)
[![FAISS](https://img.shields.io/badge/FAISS-blue?style=for-the-badge)](https://github.com/facebookresearch/faiss)

A state-of-the-art multi-agent financial assistant built with **FastAPI**, **FAISS**, and **Groq LLM**. This assistant orchestrates specialized agents to handle RAG (Retrieval-Augmented Generation), stock analysis, portfolio management, web search, and automated emailing.

**Frontend Repository:** [finance-agent-frontend](https://github.com/SemerNahdi/finance-agent-frontend)

---

## Features

- **MCP Orchestrator**: Intelligent query routing with concurrency and caching.
- **Advanced RAG**: Ingests PDFs, CSVs, and JSONs to provide context-aware financial answers.
- **Stock Intelligence**: Real-time ticker summaries and price tracking via `yfinance`.
- **Portfolio Analyzer**: Track holdings, calculate P/L, and view sector allocations.
- **Automated Emailing**: Receive daily portfolio snapshots and performance reports.
- **Web Search**: Real-time financial news integration via NewsAPI.
- **Multilingual**: Automatic language detection and dialect-aware responses.

---

## Project Structure

```text
📁 .
  📄 main.py                   # FastAPI entry point
  📁 api/
    📄 routes.py               # API endpoints 
  📁 services/
    📄 mcp_agent.py            # Central Orchestrator
    📁 agents/                 # Specialized AI Agents
      📄 rag_agent.py
      📄 stock_agent.py
      📄 portfolio_agent.py
      📄 email_agent.py
      📄 websearch_agent.py
    📁 tools/                  # Core logic & Utility tools
      📄 stock_tool.py
      📄 websearch_tool.py
      📄 groq_wrapper.py
    📁 rag/                    # RAG Implementation
      📄 embeddings.py         # FAISS & Sentence Transformers
      📄 chunking.py           # Text splitting logic
      📄 parser.py             # Document parsing (PDF, CSV, JSON)
    📁 email/                  # Emailing services
  📁 data/                     # Source documents & FAISS index
  📁 db/                       # Vector database storage
  📄 requirements.txt          # Project dependencies
```

---

##  Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/SemerNahdi/FinAgent.git
cd FinAgent
```

### 2. Create a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory and add the following keys:

```env
# --- LLM & Search ---
GROQ_API_KEY=your_groq_api_key_here
NEWS_API_KEY=your_newsapi_key_here

# --- RAG Settings ---
EMBED_MODEL=all-MiniLM-L6-v2
RAG_DATA_DIR=./data
RAG_INDEX_DIR=./data/finance_agent_index

# --- Email Settings (SMTP) ---
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
EMAIL_SENDER_NAME="FinAgent Assistant"
```

---

## Usage

### Data Ingestion (RAG)
Place your financial documents (PDF, CSV, JSON) in the `./data` directory, then run the ingestion tool:

```bash
python services/rag/rag_tool.py --ingest
```

### Start the API
Run the server using Uvicorn:

```bash
uvicorn main:app --reload
```
The API will be available at `http://localhost:8000`.

### Example Queries
You can interact with the agent via the `/api/ask` endpoint:

- **Stock**: "What is the current price of NVDA?"
- **Portfolio**: "Show me my sector allocation."
- **RAG**: "Explain the risk factors mentioned in the latest earnings report."
- **Email**: "Send the daily portfolio snapshot to my email."
- **Web**: "What are the latest headlines about interest rates?"

---

## Testing

Run the test suite using `pytest`:
```bash
pytest tests/
```


##  License

Distributed under the MIT License. See `LICENSE` for more information.

---
*Created by [Semer Nahdi](mailto:semernahdi25@gmail.com)*
