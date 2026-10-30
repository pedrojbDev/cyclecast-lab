---
license: mit
library_name: scikit-learn
pipeline_tag: tabular-regression
---
# CycleCast Lab model card

A two-hidden-layer ReLU neural network trained **from scratch** on daily total
rentals in the historical UCI Bike Sharing dataset. It forecasts one day ahead from
seven prior completed days (oldest first), plus target weekday and day-of-year cyclic
features. It has no access to actual future weather, target-day demand or rider-type
breakdowns. JSON weights and train-only scaler parameters are evaluated by the
[CycleCast Lab repository](https://github.com/pedrojbDev/cyclecast-lab), without pickle.

## Data provenance

Hadi Fanaee-T, *Bike Sharing* (2013), UCI Machine Learning Repository,
[DOI 10.24432/C5W894](https://doi.org/10.24432/C5W894),
[dataset record](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).
Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit and cite
the dataset and original work. The raw dataset is not included; code and model are
MIT licensed, which does not change the dataset license.

## Evaluation

Chronological split by **target date**: 602 training targets through 2012-08-31;
61 validation targets in September–October; 61 test targets in November–December.
Features for early validation/test dates can include prior dates in an earlier split,
which would be known at prediction time. Model and scaling are fit on training only;
the absolute validation errors set the interval radius, and the fixed test is used
only for final evaluation. No model selection on test.

Test MAE **924.62** rentals/day, RMSE **1120.41**; same-weekday-last-week baseline
MAE **1486.70**, RMSE **1968.91**. November MAE 900.32; December MAE 948.14.
The empirical interval radius is 2222.16 rentals and test coverage was 59/61 (96.72%),
with mean width 4404.98 rentals. This is a wide interval. The order statistic aims at
90% under exchangeability, but time-series residuals and distribution shifts violate
that assumption; **no future coverage guarantee**. Full numbers in metrics.json.

## Limits and intended use

The data describe one bike-share system in 2011–2012, not current demand, other
cities, or a representative distribution of events. There is no live feed, weather
forecast, holiday indicator, intervention, capacity or external validation. Counts
can change due to capacity, seasonality or operational changes. Do not use the model
for operational or safety decisions. Output can be highly uncertain; the UI displays
an empirical range, not a guaranteed prediction. See training-manifest.json for the
source snapshot, artifact SHA and exact training commit. Training was CPU-only.

Original paper: H. Fanaee-T and J. Gama, “Event labeling combining ensemble
detectors and background knowledge,” *Progress in Artificial Intelligence* (2013),
[doi:10.1007/s13748-013-0040-3](https://doi.org/10.1007/s13748-013-0040-3).
