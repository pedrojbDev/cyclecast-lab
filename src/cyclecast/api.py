"""HTTP interface for a one-day forecast and interactive demonstration."""

import json
from datetime import date
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .core import forecast

ROOT = Path(__file__).resolve().parents[2]


class ForecastRequest(BaseModel):
    date: date
    history: Annotated[
        list[Annotated[float, Field(ge=0, le=100000, allow_inf_nan=False)]],
        Field(min_length=7, max_length=7),
    ]


def create_app(model):
    app = FastAPI(title="CycleCast Lab", version="1.0.0")

    @app.get("/health")
    def health():
        return {"model_loaded": model is not None}

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(ROOT / "web" / "index.html")

    @app.get("/api/backtest")
    def backtest():
        path = ROOT / "model" / "backtest.json"
        if not path.exists():
            raise HTTPException(status_code=503, detail="Backtest unavailable")
        return json.loads(path.read_text())

    @app.post("/api/forecast")
    def prediction(request: ForecastRequest):
        if model is None:
            raise HTTPException(status_code=503, detail="Model artifact unavailable")
        return forecast(model, request.date, request.history)

    return app


def load_default():
    path = ROOT / "model" / "model.json"
    return json.loads(path.read_text()) if path.exists() else None


app = create_app(load_default())
