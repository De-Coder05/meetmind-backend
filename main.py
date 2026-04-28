from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from core.config import get_settings
from core.logging import setup_logging, get_logger
from db.database import init_db
import db.schemas
from api.routes.meetings import router as meetings_router
from api.routes.tasks import router as tasks_router

setup_logging()
logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting MeetMind backend...")
    await init_db()
    logger.info("Database initialized")
    yield
    logger.info("Shutting down MeetMind backend")


app = FastAPI(
    title="MeetMind API",
    description="Meeting accountability system — transcribe, classify, extract, push.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(meetings_router, prefix="/api/v1")
app.include_router(tasks_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "message": "MeetMind API is running",
        "docs": "/docs",
        "version": "0.1.0"
    }


@app.get("/health")
async def health():
    return {"status": "ok", "env": settings.app_env}
