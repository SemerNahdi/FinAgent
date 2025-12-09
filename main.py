# backend/main.py
from fastapi import FastAPI
from api.routes import router
from fastapi.middleware.cors import CORSMiddleware

origins = ["http://localhost:3000"]


app = FastAPI(
    title="Finance Assistant",
    description="Multi-agent financial assistant with RAG, portfolio, stock, and email tools",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Include routes
app.include_router(router, prefix="/api")
# uvicorn main:app --reload
