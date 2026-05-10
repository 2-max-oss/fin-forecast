"""FastAPI entry point for the stock forecast web app."""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from web_app.service import AnalysisError, analyze_ticker

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Stock Forecast API",
    version="1.0.0",
    description="Browser API for the stock analysis and mathematical finance model.",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/analyze/{ticker}")
async def analyze(ticker: str) -> dict:
    try:
        return await run_in_threadpool(analyze_ticker, ticker)
    except AnalysisError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

