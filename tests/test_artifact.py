import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

from cyclecast.api import app

ROOT = Path(__file__).resolve().parents[1]


def test_model_sha_and_web_page():
    artifact = ROOT / "model" / "model.json"
    manifest = json.loads((ROOT / "model" / "training-manifest.json").read_text())
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == manifest["artifact_sha256"]
    client = TestClient(app)
    page = client.get("/")
    assert page.status_code == 200
    assert "Out-of-time backtest" in page.text
    assert client.get("/health").json()["model_loaded"] is True


def test_documented_example_matches_live_response():
    request = json.loads((ROOT / "examples" / "request.json").read_text())
    expected = json.loads((ROOT / "examples" / "response.json").read_text())
    response = TestClient(app).post("/api/forecast", json=request)
    assert response.status_code == 200
    assert response.json() == expected
