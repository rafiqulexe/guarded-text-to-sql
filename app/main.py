"""FastAPI app and routes."""

from fastapi import FastAPI

from app.config import get_settings

settings = get_settings()

app = FastAPI(title="sqlguard-llm")
