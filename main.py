# backend/main.py
from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="Finance Assistant",
    description="Multi-agent financial assistant with RAG, portfolio, stock, and email tools",
    version="1.0.0",
)

# Include routes
app.include_router(router, prefix="/api")
# uvicorn main:app --reload
