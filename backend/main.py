import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router as api_router
from app.api.websockets import router as ws_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ASTRA Trading API Sandbox",
    description="SaaS Backend for Indian & US Markets AI Trading Platform",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(ws_router)

@app.get("/")
async def root():
    return {"status": "ok", "message": "ASTRA Trading API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
