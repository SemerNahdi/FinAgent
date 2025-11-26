# Finance  Agent

A multi-agent financial assistant. Built with FastAPI, FAISS, and LLM integration .

---

## Table of Contents

* [Project Structure](#project-structure)
* [Requirements](#requirements)
* [Setup](#setup)
* [Data](#data)
* [Usage](#usage)

  * [Indexing Data](#indexing-data)
  * [Querying](#querying)
* [Agents & Tools](#agents--tools)
* [Extending / Customization](#extending--customization)
* [License](#license)

---

## Project Structure

```
backend/
├─ main.py                     # FastAPI entry point
├─ api/
│   └─ routes.py               # Endpoints: /chat, /upload, /status
├─ services/
│   ├─ mcp_agent.py            # MCP agent orchestrator
│   ├─ agents/
│   │   ├─ rag_agent.py        # Orchestrates RAG tasks
│   │   ├─ stock_agent.py
│   │   ├─ portfolio_agent.py
│   │   └─ summarizer_agent.py
│   ├─ rag/                    # RAG module
│   │   ├─ embeddings.py
│   │   ├─ chunking.py
│   │   ├─ parser.py
│   │   └─ rag_tool.py
│   ├─ tools/
│   │   ├─ stock_tool.py
│   │   ├─ portfolio_tool.py
│   │   └─ summarizer_tool.py
├─ db/
│   └─ embeddings_db.py
├─ models/
│   └─ query_models.py
└─ requirements.txt
data/
└─ ...                        # PDFs, CSVs, JSONs, metadata
```

---

## Requirements

Install via:

```bash
pip install -r requirements.txt
```

---

## Setup

1. Clone the repo:

```bash
git clone https://github.com/SemerNahdi/FinAgent.git
cd Finance-agent
```

2. Place your data files in `./data`:

* PDFs (annual reports, earnings calls, etc.)
* CSVs (portfolio, metadata, etc.)
* JSON (schemas, portfolio schema)

3. Set environment variables if needed:

```bash
export RAG_DATA_DIR="./data"
export RAG_INDEX_DIR="./data/faiss_index"
export EMBED_MODEL="all-MiniLM-L6-v2"
```

---

## Usage

### Indexing Data

Run the RAG ingest script once to parse, chunk, and embed documents:

```bash
python backend/services/rag/rag_tool.py --ingest
```

* Builds FAISS index under `./data/faiss_index`
* Stores metadata for retrieval

---

## Agents & Tools

* **MCP Agent**: Orchestrates multiple agents
* **RAG Agent**: Handles document ingestion and retrieval
* **Stock Agent / Tool**: Financial stock queries
* **Portfolio Agent / Tool**: Portfolio analysis and simulation
* **Summarizer Agent / Tool**: Summarizes retrieved data

