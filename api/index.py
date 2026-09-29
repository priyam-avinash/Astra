"""
Vercel entry point — serves the FastAPI backend as one Python function.

vercel.json rewrites /api/*, /strategies/*, /brokers/*, /upstox/* and /health
here; everything else is the static React build (frontend/dist).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

os.environ.setdefault("ASTRA_SERVERLESS", "1")
os.environ.setdefault("AUTH_MODE", "jwt")          # never run the dev bypass on a public URL
os.environ.setdefault("PAPER_TRADING", "true")     # paper only — there is no live order path

from main import app  # noqa: E402,F401
