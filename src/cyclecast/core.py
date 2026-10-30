"""Causal windows and portable MLP inference."""

import math
from datetime import date, timedelta

import numpy as np


def make_features(day, history):
    if len(history) != 7 or any(not math.isfinite(float(n)) or n < 0 for n in history):
        raise ValueError("Seven non-negative finite observations required")
    weekday = 2 * math.pi * day.weekday() / 7
    year = 2 * math.pi * (day.timetuple().tm_yday - 1) / 365.25
    return list(history) + [math.sin(weekday), math.cos(weekday), math.sin(year), math.cos(year)]


def build_windows(rows):
    if any(rows[i][0] != rows[i - 1][0] + timedelta(days=1) for i in range(1, len(rows))):
        raise ValueError("Dates must be consecutive, unique and ordered")
    return [
        (rows[i][0], make_features(rows[i][0], [r[1] for r in rows[i - 7 : i]]), rows[i][1])
        for i in range(7, len(rows))
    ]


def split_windows(windows):
    return (
        [w for w in windows if w[0] <= date(2012, 8, 31)],
        [w for w in windows if date(2012, 9, 1) <= w[0] <= date(2012, 10, 31)],
        [w for w in windows if date(2012, 11, 1) <= w[0] <= date(2012, 12, 31)],
    )


def export_model(sx, sy, nn, interval):
    return {
        "version": 1,
        "feature_mean": sx.mean_.tolist(),
        "feature_scale": sx.scale_.tolist(),
        "target_mean": float(sy.mean_[0]),
        "target_scale": float(sy.scale_[0]),
        "weights": [w.tolist() for w in nn.coefs_],
        "biases": [b.tolist() for b in nn.intercepts_],
        "interval_radius": float(interval),
    }


def forecast(model, day, history):
    x = np.asarray(make_features(day, history), dtype=float)
    x = (x - np.asarray(model["feature_mean"])) / np.asarray(model["feature_scale"])
    weights, biases = model["weights"], model["biases"]
    for i, (weight, bias) in enumerate(zip(weights, biases, strict=True)):
        x = x @ np.asarray(weight) + np.asarray(bias)
        if i < len(weights) - 1:
            x = np.maximum(x, 0)
    value = max(0.0, float(x[0] * model["target_scale"] + model["target_mean"]))
    radius = float(model["interval_radius"])
    return {
        "date": day.isoformat(),
        "prediction": value,
        "lower": max(0.0, value - radius),
        "upper": value + radius,
        "baseline": float(history[0]),
    }
