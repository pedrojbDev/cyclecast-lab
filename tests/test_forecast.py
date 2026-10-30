from datetime import date, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from cyclecast.api import create_app
from cyclecast.core import build_windows, export_model, forecast, make_features, split_windows


def test_windows_use_only_seven_prior_days():
    start = date(2012, 8, 25)
    rows = [(start + timedelta(days=i), i * 10) for i in range(15)]
    windows = build_windows(rows)
    assert windows[0][0] == start + timedelta(days=7)
    assert windows[0][1][:7] == [0, 10, 20, 30, 40, 50, 60]
    assert windows[0][2] == 70
    train, val, test = split_windows(windows)
    assert not train
    assert all(date(2012, 9, 1) <= w[0] <= date(2012, 10, 31) for w in val)
    assert not test


def test_missing_date_fails_closed():
    rows = [(date(2012, 1, 1) + timedelta(days=i), i) for i in range(10) if i != 5]
    with pytest.raises(ValueError, match="consecutive"):
        build_windows(rows)


def fixture_model():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1000, size=(100, 11))
    y = rng.uniform(500, 1500, size=(100, 1))
    sx, sy = StandardScaler().fit(x), StandardScaler().fit(y)
    nn = MLPRegressor(hidden_layer_sizes=(8, 4), max_iter=30, random_state=42).fit(
        sx.transform(x), sy.transform(y).ravel()
    )
    return sx, sy, nn, export_model(sx, sy, nn, 100.0)


def test_portable_model_matches_sklearn():
    sx, sy, nn, model = fixture_model()
    day, history = date(2012, 12, 1), [100, 200, 300, 400, 500, 600, 700]
    x = np.asarray([make_features(day, history)])
    expected = max(0, float(sy.inverse_transform(nn.predict(sx.transform(x)).reshape(-1, 1))[0, 0]))
    result = forecast(model, day, history)
    assert result["prediction"] == pytest.approx(expected)
    assert result["lower"] == pytest.approx(max(0, expected - 100))
    assert result["baseline"] == 100


def test_api_validates_and_forecasts():
    _, _, _, model = fixture_model()
    client = TestClient(create_app(model))
    response = client.post("/api/forecast", json={"date": "2012-12-01", "history": [100] * 7})
    assert response.status_code == 200
    assert response.json()["lower"] <= response.json()["prediction"] <= response.json()["upper"]
    assert (
        client.post("/api/forecast", json={"date": "2012-12-01", "history": [1]}).status_code == 422
    )
    assert (
        client.post("/api/forecast", json={"date": "2012-12-01", "history": [-1] * 7}).status_code
        == 422
    )
    assert (
        TestClient(create_app(None))
        .post("/api/forecast", json={"date": "2012-12-01", "history": [1] * 7})
        .status_code
        == 503
    )


def test_dashboard_exposes_aggregated_backtest():
    client = TestClient(create_app(None))
    response = client.get("/api/backtest")
    assert response.status_code == 200
    records = response.json()
    assert len(records) == 9
    assert records[0]["start"] == "2012-11-01"
    assert {"actual_mean", "forecast_mean", "baseline_mean"} <= records[0].keys()
