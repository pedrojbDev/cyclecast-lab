# CycleCast Lab

A small but complete forecasting experiment: **given the previous seven days of
bike-share rentals, estimate tomorrow's demand**. It pairs a from-scratch neural
network with a chronological backtest, a simple last-week baseline, empirical
uncertainty intervals and an interactive dashboard. This is a historical research
demo, not a live city forecasting service.

## Try it

Requires Python 3.11–3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --locked
uv run uvicorn cyclecast.api:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/** for the dashboard. It shows measured weekly-mean
backtest curves and lets you edit seven hypothetical daily counts. The model file is
included, so no dataset download is needed for inference. The chart is a compact
aggregate of the test period, not a live feed. The test metrics are calculated on
daily predictions; points are grouped into weeks only for the chart.

Or call the API:

```sh
curl -s -X POST http://127.0.0.1:8000/api/forecast \
  -H 'Content-Type: application/json' --data @examples/request.json
```

Example request (oldest to newest):

```json
{"date":"2012-12-20","history":[4800,5200,5500,5300,6000,4200,3900]}
```

Actual response (rounded here; [full response](examples/response.json) is checked in):

```json
{"date":"2012-12-20","prediction":4531.72,"lower":2309.56,"upper":6753.88,"baseline":4800.0}
```

`POST /api/forecast` requires an ISO date and exactly seven finite numbers in
[0, 100000], one per preceding day, oldest first. `baseline` is the oldest value,
the same weekday a week earlier. The model does not validate whether inputs come
from the same system. Invalid input returns 422, missing model returns 503;
`GET /health` shows model availability, `GET /api/backtest` returns aggregated
test results, and `/docs` serves OpenAPI. The endpoint accepts arbitrary dates but
**has not been validated for dates outside the historical test period**.

## How it works

UCI daily CSV → verify SHA-256 → construct strictly causal seven-day windows →
chronological split → train-only input/target standardization → ReLU MLP →
validation residual interval → JSON weights → NumPy inference → FastAPI → dashboard.
The model uses 7 lagged counts, weekday sine/cosine and day-of-year sine/cosine.
It uses neither future weather nor target-day counts. No unsafe pickle is loaded.

| Target period | Role | Days |
| --- | --- | ---: |
| 2011-01-08 through 2012-08-31 | Train | 602 |
| 2012-09-01 through 2012-10-31 | Validate and calibrate interval | 61 |
| 2012-11-01 through 2012-12-31 | Final test | 61 |

Validation and test features can use prior observations from an earlier split: they
are past information available when the forecast would be made. Model architecture
and hyperparameters were fixed before the test. The baseline repeats the observation
from seven days before the target. Splitting is by target date, not random rows.

## Results

| Metric | Neural model | Same weekday last week |
| --- | ---: | ---: |
| Test MAE (rentals/day) | **924.62** | 1486.70 |
| Test RMSE (rentals/day) | **1120.41** | 1968.91 |

November model MAE: 900.32; December: 948.14. The 90% nominal empirical interval
uses the `ceil((n+1)*0.9)`-th ordered absolute validation residual: radius 2222.16.
On the 61 test days it covered 59 (96.72%); mean interval width was 4404.98 rentals.
Coverage is descriptive, not a promise: time-series residuals are not exchangeable
and the interval is wide. Results, including error by month, are in
[metrics.json](model/metrics.json); provenance and hash are in
[training-manifest.json](model/training-manifest.json).

## Reproduce

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python -m cyclecast.training \
  --training-commit "$(git rev-parse HEAD)" --output /tmp/cyclecast-model
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest -q
```

The downloader uses a pinned 280 KB UCI archive and fails on checksum mismatch;
raw data are kept in ignored `data/`. The original recorded training code commit is
in the manifest. Different numerical libraries/platforms can alter the model hash.
On the recorded Linux CPU run, fitting and evaluation took 0.179 s with peak RSS
120,696 KiB (local cached archive, no download); timing is machine-specific.

## Limitations and data rights

Only one Washington, DC bike-share system from 2011–2012 is represented. Weather,
holidays, capacity changes and major events are not inputs; changes in mobility can
make historical patterns misleading. The interval is not a guarantee and neither
model nor dashboard is validated for current or operational use. See the
[model card](model/README.md) for details.

Hadi Fanaee-T, *Bike Sharing* (2013), UCI Machine Learning Repository,
[DOI 10.24432/C5W894](https://doi.org/10.24432/C5W894), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit the dataset creator
and cite the original paper. Raw data are not redistributed. Code and model are under
the [MIT License](LICENSE); dataset attribution requirements remain applicable.

Original paper: H. Fanaee-T and J. Gama, “Event labeling combining ensemble
detectors and background knowledge,” *Progress in Artificial Intelligence* (2013),
[doi:10.1007/s13748-013-0040-3](https://doi.org/10.1007/s13748-013-0040-3).
