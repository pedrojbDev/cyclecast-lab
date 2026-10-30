"""Train and evaluate one fixed causal daily demand model."""

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import resource
import time
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .core import build_windows, export_model, forecast, split_windows

URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
SHA256 = "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401"


def read_archive(path):
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != SHA256:
        raise ValueError("Dataset checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        text = archive.read("day.csv").decode("utf-8")
    return [
        (date.fromisoformat(row["dteday"]), int(row["cnt"]))
        for row in csv.DictReader(io.StringIO(text))
    ]


def calibrate(errors, level):
    values = np.sort(np.asarray(errors, dtype=float))
    if len(values) == 0 or not 0 < level < 1 or not np.isfinite(values).all():
        raise ValueError("Invalid calibration residuals or level")
    k = min(len(values), math.ceil((len(values) + 1) * level))
    return float(values[k - 1])


def evaluate(y, predictions, baseline, radius):
    y, pred, base = (np.asarray(x, dtype=float) for x in (y, predictions, baseline))
    errors = np.abs(y - pred)
    return {
        "n": len(y),
        "mae": float(errors.mean()),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "baseline_mae": float(np.abs(y - base).mean()),
        "baseline_rmse": float(np.sqrt(np.mean((y - base) ** 2))),
        "coverage": float(np.mean(errors <= radius)),
        "mean_interval_width": float(np.mean(pred + radius - np.maximum(0, pred - radius))),
    }


def arrays(windows):
    return np.asarray([w[1] for w in windows]), np.asarray([w[2] for w in windows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/bike.zip"))
    parser.add_argument("--output", type=Path, default=Path("model"))
    parser.add_argument("--training-commit", required=True)
    args = parser.parse_args()
    start = time.monotonic()
    if not args.archive.exists():
        with urllib.request.urlopen(URL, timeout=60) as response:
            data = response.read(10_000_001)
        if len(data) > 10_000_000:
            raise ValueError("Archive exceeds size limit")
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        args.archive.write_bytes(data)
    train, val, test = split_windows(build_windows(read_archive(args.archive)))
    x_train, y_train = arrays(train)
    x_val, y_val = arrays(val)
    sx = StandardScaler().fit(x_train)
    sy = StandardScaler().fit(y_train.reshape(-1, 1))
    nn = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.01,
        max_iter=1000,
        tol=1e-5,
        random_state=42,
    )
    nn.fit(sx.transform(x_train), sy.transform(y_train.reshape(-1, 1)).ravel())
    val_pred = np.maximum(
        0, sy.inverse_transform(nn.predict(sx.transform(x_val)).reshape(-1, 1)).ravel()
    )
    radius = calibrate(np.abs(y_val - val_pred), 0.9)
    model = export_model(sx, sy, nn, radius)

    def run(windows):
        results = [forecast(model, d, f[:7]) for d, f, _ in windows]
        y = [w[2] for w in windows]
        return evaluate(
            y, [r["prediction"] for r in results], [r["baseline"] for r in results], radius
        ), results

    val_metrics, _ = run(val)
    test_metrics, predictions = run(test)
    monthly = {}
    for month in sorted({d.strftime("%Y-%m") for d, _, _ in test}):
        mask = [i for i, (d, _, _) in enumerate(test) if d.strftime("%Y-%m") == month]
        monthly[month] = evaluate(
            [test[i][2] for i in mask],
            [predictions[i]["prediction"] for i in mask],
            [predictions[i]["baseline"] for i in mask],
            radius,
        )
    # Weekly means for a compact visual evaluation without redistributing raw rows.
    chart = []
    for i in range(0, len(test), 7):
        chunk, pred = test[i : i + 7], predictions[i : i + 7]
        chart.append(
            {
                "start": chunk[0][0].isoformat(),
                "n": len(chunk),
                "actual_mean": float(np.mean([w[2] for w in chunk])),
                "forecast_mean": float(np.mean([p["prediction"] for p in pred])),
                "baseline_mean": float(np.mean([p["baseline"] for p in pred])),
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    artifact = args.output / "model.json"
    artifact.write_text(json.dumps(model, indent=2) + "\n")
    metrics = {
        "seed": 42,
        "train_days": len(train),
        "validation": val_metrics,
        "test": test_metrics,
        "test_months": monthly,
        "interval_radius": radius,
        "fit_iterations": int(nn.n_iter_),
        "seconds": time.monotonic() - start,
        "peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.output / "backtest.json").write_text(json.dumps(chart, indent=2) + "\n")
    manifest = {
        "training_type": "from-scratch",
        "dataset_license": "CC-BY-4.0",
        "dataset": "UCI Bike Sharing, 2013, DOI:10.24432/C5W894",
        "dataset_url": URL,
        "dataset_archive_sha256": SHA256,
        "training_commit": args.training_commit,
        "split_method": "chronological target date: train through 2012-08-31, validation Sep-Oct, test Nov-Dec 2012",
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "model": "MLPRegressor (64,32), ReLU, seed 42",
        "python": platform.python_version(),
    }
    (args.output / "training-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
