# STTL — Dengue Weekly-Cases Forecasting (Python)

Predict the **weekly number of dengue fever cases** (`total_cases`) for two
tropical cities — **San Juan (`sj`)** and **Iquitos (`iq`)** — from historical
weather, climate and vegetation data (the [DengAI](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/)
dataset).

This is a production-quality Python reimplementation inspired by the methodology
of the R analysis at <https://rpubs.com/jencheng/dengue>. It reproduces the
*workflow* (EDA → cleaning → feature engineering → modeling → evaluation →
visualization), **not** a line-by-line translation, and restricts modeling to a
specified set of regression models.

---

## Key design decisions

| Decision | Why |
| :--- | :--- |
| **Two cities modeled independently** | San Juan and Iquitos have different climates and dengue dynamics (confirmed by feature-selection overlap); one global model would blur both. |
| **Drop `reanalysis_sat_precip_amt_mm`** | It is byte-identical to `precipitation_amt_mm` — a known DengAI redundancy that would double-count one signal. |
| **Past-only imputation (ffill → bfill)** | Weather is temporally smooth; the best estimate of a missing week is the most recent observed week. Causal — never fills from the future (a single `bfill` only handles leading gaps). |
| **Lag (1/2/3/4 wk) + rolling (4/8/12 wk) + seasonality features** | Encodes the ~3–6 week biological delay between weather and transmission, denoises weather, and captures dengue's strong annual cycle. |
| **`log1p(total_cases)` target** | The raw target is heavily right-skewed by rare outbreak spikes; log-transform makes it near-symmetric so models optimise evenly. Predictions are inverted with `expm1`. |
| **Temporal hold-out (earliest 75% train / latest 25% test)** | Mirrors the real deployment scenario for a short time series; no shuffling, so no future leakage. |
| **`SelectKBest(mutual_info_regression, k=30)`, fit on train only** | MI captures non-linear climate→mosquito relationships that Pearson misses, and a filter method gives every model the identical feature set for a fair comparison. |
| **`StandardScaler` fit on train only** | Needed by the linear family; harmless to trees. Fitting on train alone prevents test-set leakage. |
| **Tune only tree models** | Linear/Ridge/Lasso/ElasticNet are strong, cheap baselines with fixed hyperparameters; tree capacity is highly sensitive and benefits from search. |

---

## Models

Exactly **eight** regression models are implemented (and no others):

| Model | Family | Hyperparameter tuning |
| :--- | :--- | :--- |
| Linear Regression | linear | — (untuned) |
| Ridge (L2) | linear | — (untuned) |
| Lasso (L1) | linear | — (untuned) |
| Elastic Net (L1+L2) | linear | — (untuned) |
| Decision Tree | tree | **GridSearchCV** (small, exhaustive) |
| Random Forest | ensemble | **RandomizedSearchCV** (large space) |
| XGBoost | boosting | **RandomizedSearchCV** |
| LightGBM | boosting | **RandomizedSearchCV** |

**Why these search strategies?** The Decision Tree has few impactful knobs with a
small discrete set of sensible values, so an exhaustive grid is cheap and optimal.
Random Forest / XGBoost / LightGBM have large joint spaces where a full grid is
prohibitive; `RandomizedSearchCV` samples configurations and reliably finds
near-optimal settings at a fraction of the cost (Bergstra & Bengio, 2012). All
searches use `TimeSeriesSplit` (validation folds strictly follow training folds)
scored by **negative MAE** — the official DengAI metric.

---

## Evaluation

Every model is scored on the temporal hold-out with **MAE** (primary), **RMSE**
and **R²**, presented as a per-city comparison table and ranked best→worst, plus
a combined (both-cities-pooled) DengAI-style score. Reports are written to
`outputs/reports/`.

---

## Project structure

```
STTL Project/
├── config/
│   └── config.yaml                # every tunable knob (paths, splits, grids, styling)
├── dataset/                       # DengAI CSVs (provided)
├── dengue_forecast/               # the package
│   ├── config.py                  # typed YAML config loader
│   ├── utils.py                   # logging, seeding, IO helpers
│   ├── data_loader.py             # load + merge CSVs, per-city split
│   ├── preprocessing.py           # cleaning, imputation, scaling
│   ├── feature_engineering.py     # lags, rolling, seasonality, target transform, selection
│   ├── models.py                  # factory for the 8 permitted models
│   ├── tuning.py                  # Grid/Randomized search (tree models)
│   ├── evaluation.py              # MAE/RMSE/R², comparison & ranking
│   ├── visualization.py           # all 300-DPI publication figures
│   └── pipeline.py                # end-to-end orchestrator
├── scripts/
│   ├── run_pipeline.py            # CLI entry point (weather-only forecast)
│   └── run_autoregressive.py      # opt-in nowcast run -> outputs_autoregressive/
├── tests/                         # pytest suite (config, data, leakage, eval, integration)
├── outputs/                       # generated: figures/, models/, reports/, predictions/
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. (recommended) create a virtual environment
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt
```

Requires Python 3.9+.

## Usage

```bash
# Run the whole pipeline (data → features → 8 models → tuning → eval → figures)
python scripts/run_pipeline.py

# Optionally point at a custom config
python scripts/run_pipeline.py --config config/config.yaml

# Autoregressive (nowcast) mode — adds past-case features for much lower RMSE.
# Writes to outputs_autoregressive/ (leaves the weather-only outputs/ intact).
python scripts/run_autoregressive.py
```

### Two modes: forecast vs. nowcast

| Mode | Command | Uses past cases? | Hold-out RMSE (RF/XGB) | Question answered |
| :--- | :--- | :--- | :--- | :--- |
| **Weather-only forecast** (default) | `run_pipeline.py` | No | ~27 (SJ) / ~13 (IQ) | DengAI-legit multi-week forecast from weather alone |
| **Autoregressive nowcast** (opt-in) | `run_autoregressive.py` | Yes (lags of `total_cases`) | ~11 (SJ) / ~9 (IQ) | 1-week-ahead operational nowcast |

Weekly dengue cases are strongly autocorrelated, so feeding the models recent
case counts (lags 1–4 weeks + a 4-week rolling mean) cuts RMSE ~2.5× and lifts R²
from ~0 to 0.83–0.87 (San Juan). **Caveat:** this assumes recent case counts are
known at prediction time — it is a shorter-horizon *nowcast*, not the pure
weather forecast, and the test set is forecast recursively (error compounds over
the horizon). Weather-only remains the honest default.

Or drive it from Python:

```python
from dengue_forecast.pipeline import DenguePipeline
outputs = DenguePipeline().run()
print(outputs["report"])
```

---

## Outputs

After a run, `outputs/` contains:

| Path | Contents |
| :--- | :--- |
| `reports/all_metrics.csv` | Every model × city: MAE / RMSE / R². |
| `reports/ranked_metrics.csv` | The same, ranked best→worst within each city. |
| `reports/combined_scores.csv` | Both cities pooled into one score per model (DengAI-style). |
| `reports/best_models.csv` | The winning model per city. |
| `reports/model_comparison_report.txt` | Human-readable comparison report. |
| `reports/run_summary.json` | Machine-readable run summary. |
| `predictions/{city}_{model}_holdout.csv` | Hold-out actual-vs-predicted per model. |
| `predictions/submission.csv` | 416-week test forecast (best model per city). |
| `models/{city}/{model}.joblib` | Every trained hold-out estimator. |
| `figures/` | All figures (see below), 300 DPI. |

### Figures (all 300 DPI, consistent styling)

* **Global / EDA:** `missing_values.png`, `target_overview.png`,
  `model_comparison.png`, per-city `eda/correlation_heatmap.png`,
  `eda/feature_selection_mi.png`, `<city>/all_models_timeline.png`.
* **Per model × per city** (`figures/<city>/<model>/`):
  `actual_vs_predicted.png`, `residuals_over_time.png`, `error_scatter.png`,
  `residual_hist.png`, `feature_importance.png` (tree models),
  and a combined 6-panel `dashboard.png`.

---

## Testing

```bash
python -m pytest          # 22 tests, ~11 s
```

The suite (`tests/`) guards config loading, the exactly-eight-model constraint,
data integrity (row counts, chronological order, zero NaNs after cleaning/feature
engineering), **leakage** (every hold-out week is strictly after every train
week; scaler/selector are train-fit), the `log1p`↔`expm1` round-trip, the
evaluation metrics/ranking, and a fast end-to-end run through the real pipeline
stack (plus both search paths). No full tuning sweep is triggered, so it stays
fast.

## Reproducibility

A single global seed (`project.random_seed`, default `42`) seeds Python, NumPy
and every estimator, so repeated runs produce identical results. All randomness
in feature selection (MI estimator) and tuning (`RandomizedSearchCV`) is seeded.

## Limitations & future work

1. **Weather glass ceiling.** Outbreak **peaks are systematically
   under-predicted** — weather alone cannot explain the magnitude of the rarest
   large outbreaks (unmeasured human/immunological drivers). MAE is near the
   practical floor for weather-only inputs and R² is intrinsically low for spiky
   counts, so **MAE, not R², is the meaningful metric.** The autoregressive mode
   is the one lever that helps, at the cost of reframing the task as a nowcast.
2. **Single temporal hold-out** (75/25), not rolling-origin CV — defensible for a
   short series and matches deployment, but walk-forward validation would tighten
   the variance estimates. *Future work.*
3. **R-notebook fidelity.** The reference rpubs page is JavaScript-rendered and
   could not be scraped, so this reproduces the standard DengAI workflow and
   analysis *style* — it is inspired by, not a literal translation of, the R work.
4. **Autoregressive submission is a nowcast**, forecast recursively (error
   compounds over the horizon), so it is not a DengAI-valid multi-week entry; the
   weather-only `outputs/` is the competition-legitimate deliverable.

---

## License

Released under the [MIT License](LICENSE).
