# STTL Dengue Forecasting — Project Summary

**Goal:** predict the weekly number of dengue fever cases (`total_cases`) for two
tropical cities — **San Juan (`sj`)** and **Iquitos (`iq`)** — from historical
weather, climate and vegetation data (the DengAI dataset). This is a supervised
**time-series regression** problem, and the two cities are modeled
**independently** because their climates and dengue dynamics differ sharply.

The implementation is a modular, reproducible Python package
(`dengue_forecast/`) driven by a single YAML config, executed via
`python scripts/run_pipeline.py`. Every run regenerates all tables and figures
under `outputs/`. This document is the running record of the pipeline; each step
is followed by the concrete outcomes (files and figures) it produces.

---

## Dataset at a glance

| Item | Detail |
| :--- | :--- |
| Training weeks | 1,456 (936 San Juan + 520 Iquitos) |
| Test weeks | 416 (260 San Juan + 156 Iquitos) |
| San Juan span | 1990-04-30 → 2008-04-22 |
| Iquitos span | 2000-07-01 → 2010-06-25 |
| Raw predictive features | 20 provided → **19 used** (1 redundant reanalysis duplicate dropped): 4 vegetation NDVI, 1 rainfall, 9 atmospheric-reanalysis, 5 weather-station |
| Engineered features | **213 per city** (19 base + lags + rolling stats + seasonality) → **30 selected** by mutual information |
| Target | `total_cases` (weekly dengue count; right-skewed) |
| Key columns | `city`, `year`, `weekofyear`, `week_start_date` |

---

## Repository & module layout

```
STTL Project/
├── config/config.yaml              # every knob: paths, split, lags, tuning grids, styling
├── dataset/                        # the provided DengAI CSVs
├── dengue_forecast/                # the package (typed, docstringed, PEP-8, logged)
│   ├── config.py    utils.py       # YAML config loader; logging/seeding/helpers
│   ├── data_loader.py              # load + merge CSVs, per-city split
│   ├── preprocessing.py            # drop redundancy, ffill→bfill impute, StandardScaler
│   ├── feature_engineering.py      # lags + rolling + seasonality, log1p, MI SelectKBest
│   ├── models.py                   # factory for the 8 permitted models only
│   ├── tuning.py                   # Grid (DT) + Randomized (RF/XGB/LGBM) search
│   ├── evaluation.py               # MAE/RMSE/R², ranking, combined score
│   ├── visualization.py            # all figures @ 300 DPI, consistent styling
│   └── pipeline.py                 # end-to-end orchestrator
├── scripts/
│   ├── run_pipeline.py             # weather-only forecast (default)
│   ├── run_autoregressive.py       # full nowcast run -> outputs_autoregressive/
│   └── nowcast_demo.py             # climate-only vs climate+cases head-to-head
├── tests/                          # pytest suite (config, data, leakage, eval, integration)
├── outputs/                        # figures/, reports/, predictions/, models/, nowcast_demo/
├── pytest.ini  requirements.txt  README.md  .gitignore
```

**Allowed models (exactly eight; nothing else):** Linear Regression, Ridge (L2),
Lasso (L1), Elastic Net (L1+L2), Decision Tree, Random Forest, XGBoost, LightGBM.
No SVM, KNN, neural nets, GAM, Bayesian, or ensembles beyond this list.

---

## STEP 1 — Data loading & per-city separation

**What we did.** Read the three DengAI CSVs; merged features with labels on the
composite key `(city, year, weekofyear)` (one-to-one, no rows dropped); parsed
`week_start_date` to a real datetime; sorted each city chronologically; and split
into the two independently-modeled cities.

**What we found.** 936 San Juan + 520 Iquitos training weeks and 260 + 156 test
weeks, spanning the ranges listed above.

### Outcomes

* In-memory per-city frames (train with target, test without). No files written
  at this stage; the cleaned versions are produced in Step 2.

---

## STEP 2 — Data cleaning & missing-value imputation

**What we did**, per city in chronological order (so no future information leaks
backward):

1. **Dropped one redundant column** — `reanalysis_sat_precip_amt_mm` is
   byte-for-byte identical to `precipitation_amt_mm`, so keeping it would
   double-count one signal. This leaves **19** weather predictors.
2. **Past-only imputation** — forward-fill (carry the last valid observation
   forward), then a single back-fill pass for the few *leading* gaps that have no
   past to draw from. Weather is temporally smooth, so the most recent observed
   week is a low-error estimate, and the fill is causal.

**What we found.**

| Group | Missing cells before | Worst offenders | After impute |
| :--- | ---: | :--- | ---: |
| San Juan train | 371 | `ndvi_ne` 191, `ndvi_nw` 49, `ndvi_se` 19 | **0** |
| Iquitos train | 164 | `station_diur_temp_rng_c` 37, `station_avg_temp_c` 37, `station_precip_mm` 16 | **0** |

The two cities miss **different** things — San Juan's gaps are in **satellite
NDVI** (vegetation), Iquitos's are in **ground weather-station** readings
(instrument outages) — which is direct evidence for cleaning and modeling each
city separately. No corrupt values were present (NDVI within [-1, 1], no negative
rainfall/cases, humidity within [0, 100]%).

### Outcomes

* Cleaned, fully-imputed per-city frames carried forward in memory.
* **Figure:** `outputs/figures/missing_values.png` (see the Visualization
  Catalogue in Step 7).

---

## STEP 3 — Feature engineering (lags, rolling, seasonality)

**What we did**, for each city independently and **causally** (only ever looking
backward in time). Lags/rolling are built on the train and test series **joined
in date order**, so the earliest test weeks legitimately inherit their history
from the tail of train (a lag only looks backward — no leakage):

1. **Lag features** — each of the 19 weather variables shifted back by
   **1, 2, 3, 4 weeks**, encoding the ~3–6 week biological delay between weather
   and transmission (mosquito breeding → larval maturation → biting → incubation).
2. **Rolling statistics** — trailing **mean and std over 4/8/12-week windows**
   (shifted by 1 so the window is strictly trailing), which denoise weather —
   mosquito populations respond to *sustained* conditions, not one-week blips.
3. **Seasonality** — cyclic `sin`/`cos` encodings of week-of-year and month.
   Calendar position is known for any future date, so this is *not* leakage and
   captures dengue's strong annual cycle.

This yields **213 candidate features per city**. Rows with undefined lag/rolling
values (the **first 6 training weeks** of each city) are dropped from *train
only*; **all test rows are retained** because every test week must remain
predictable.

**What we found.** Train matrices reduce to **930 rows (San Juan)** and **514
rows (Iquitos)**; test stays at 260 and 156.

### Outcomes

* Engineered feature matrices (`X_train`, `X_test`) per city, in memory.

---

## STEP 4 — Target transform, feature selection & scaling

**What we did**, all fit on the **training portion only** (leakage-safe):

1. **Target transform** — train on `y = log1p(total_cases)`; predictions are
   inverted with `expm1` and clipped at 0. The raw target is heavily
   right-skewed by rare outbreak spikes, and the log makes it near-symmetric so
   models optimise evenly.
2. **Feature selection** — `SelectKBest(mutual_info_regression, k=30)`, fit on
   the training portion against the log target. Mutual information is preferred
   over Pearson correlation because climate→mosquito relationships are non-linear
   and threshold-like; MI captures *any* dependency and is scale-free. As a
   **filter** method it hands every model the identical 30-feature set, making
   the comparison a fair test of the *estimator*. Reduces **213 → 30** per city.
3. **Scaling** — `StandardScaler` fit on the training portion only, then applied
   to hold-out and test. Needed by the linear family; harmless to trees.

**What we found.** The two cities select **different** features (San Juan favours
longer-lag/rolling temperature-humidity; Iquitos favours shorter lags),
reinforcing the per-city design.

### Outcomes

* Fitted selector + scaler per city (reused for the final full-data refit).
* **Figures:** `outputs/figures/<city>/eda/feature_selection_mi.png` and
  `correlation_heatmap.png` (Step 7).

---

## STEP 5 — Train/test split, models & hyperparameter tuning

**Temporal hold-out.** Each city's engineered training matrix is split by time —
**earliest 75% = train, most-recent 25% = hold-out test** — with no shuffling, so
no future leakage. This mirrors the real deployment scenario for a short series.

**The eight models.** All share the identical selected + scaled feature matrix;
only the estimator changes.

| Model | Family | Tuned? | Search |
| :--- | :--- | :--- | :--- |
| Linear Regression | linear | no | — |
| Ridge (L2) | linear | no | — |
| Lasso (L1) | linear | no | — |
| Elastic Net (L1+L2) | linear | no | — |
| Decision Tree | tree | **yes** | GridSearchCV (72 combinations) |
| Random Forest | ensemble | **yes** | RandomizedSearchCV (40 samples) |
| XGBoost | boosting | **yes** | RandomizedSearchCV (40 samples) |
| LightGBM | boosting | **yes** | RandomizedSearchCV (40 samples) |

**Why these search strategies.** The Decision Tree has few impactful knobs
(`max_depth`, `min_samples_leaf`, `min_samples_split`) over a small discrete set,
so an exhaustive **grid** is cheap and guaranteed-best-in-grid. Random Forest /
XGBoost / LightGBM have large joint spaces where a full grid is prohibitive, so
**randomized search** samples 40 configurations and reliably finds near-optimal
settings at a fraction of the cost (Bergstra & Bengio, 2012). The linear family
is intentionally **not** tuned — it serves as a strong, cheap, interpretable
baseline with fixed hyperparameters from the config.

**Cross-validation.** All searches use `TimeSeriesSplit` (`n_splits=4`) so every
validation fold is strictly *after* its training fold, scored by **negative MAE**
— the official DengAI metric.

**Tuning objective (configurable).** `cv.tune_objective` selects what the search
minimizes: `"mae"` (default, the official metric) or `"rmse"` (RMSE on the
original case scale — predictions are inverse-transformed inside each CV fold
before scoring). Experiments show RMSE-tuning yields only a **small** RMSE gain
in weather-only mode (e.g. San Juan Random Forest 27.3 → 26.8) and can slightly
**hurt** autoregressive mode (San Juan XGBoost 11.5 → 12.9), confirming the
models already sit near their error floor. MAE is kept as the default.

### Outcomes — `outputs/models/<city>/`

* One trained hold-out estimator per model/city (`*.joblib`, 16 files total).

---

## STEP 6 — Evaluation & results

Every model is scored on the temporal hold-out with **MAE (primary)**, **RMSE**
and **R²**, ranked best→worst per city, plus a combined (both-cities-pooled)
DengAI-style score. The best model per city is then **refit on 100% of training
weeks** and forecasts the 416-week test set.

**San Juan — hold-out (ranked by MAE).**

| Rank | Model | MAE | RMSE | R² | Tuned |
| ---: | :--- | ---: | ---: | ---: | :--- |
| 1 | **Elastic Net (L1+L2)** | **16.992** | 25.977 | 0.174 | no |
| 2 | Lasso (L1) | 17.085 | 25.969 | 0.175 | no |
| 3 | Random Forest | 17.209 | 27.289 | 0.089 | yes |
| 4 | XGBoost | 17.760 | 27.761 | 0.057 | yes |
| 5 | Ridge (L2) | 18.097 | 27.499 | 0.075 | no |
| 6 | Linear Regression | 18.376 | 28.092 | 0.034 | no |
| 7 | LightGBM | 18.759 | 27.427 | 0.079 | yes |
| 8 | Decision Tree | 18.954 | 31.332 | −0.201 | yes |

**Iquitos — hold-out (ranked by MAE).**

| Rank | Model | MAE | RMSE | R² | Tuned |
| ---: | :--- | ---: | ---: | ---: | :--- |
| 1 | **XGBoost** | **7.421** | 12.456 | 0.007 | yes |
| 2 | Random Forest | 7.622 | 13.234 | −0.120 | yes |
| 3 | LightGBM | 7.636 | 13.019 | −0.085 | yes |
| 4 | Decision Tree | 7.905 | 12.381 | 0.019 | yes |
| 5 | Linear Regression | 8.020 | 13.206 | −0.116 | no |
| 6 | Lasso (L1) | 8.022 | 13.351 | −0.140 | no |
| 7 | Elastic Net (L1+L2) | 8.058 | 13.318 | −0.135 | no |
| 8 | Ridge (L2) | 8.099 | 13.245 | −0.122 | no |

**Combined (both cities pooled, DengAI-style; ranked by MAE).**

| Rank | Model | MAE | RMSE | R² |
| ---: | :--- | ---: | ---: | ---: |
| 1 | **Random Forest** | **13.793** | 23.275 | 0.122 |
| 2 | Elastic Net (L1+L2) | 13.808 | 22.306 | 0.194 |
| 3 | Lasso (L1) | 13.855 | 22.306 | 0.194 |
| 4 | XGBoost | 14.076 | 23.480 | 0.106 |
| 5 | Ridge (L2) | 14.534 | 23.436 | 0.110 |
| 6 | Linear Regression | 14.685 | 23.876 | 0.076 |
| 7 | LightGBM | 14.795 | 23.337 | 0.117 |
| 8 | Decision Tree | 15.016 | 26.201 | −0.113 |

**Best per city:** **Elastic Net** (San Juan) and **XGBoost** (Iquitos).
**Overall best by combined MAE:** **Random Forest** — which is exactly why we
benchmark all models and pick per city rather than assuming one wins. The top
few models sit within ~0.3–1 MAE of each other.

### Outcomes — `outputs/reports/` and `outputs/predictions/`

| File | Description |
| :--- | :--- |
| `reports/all_metrics.csv` | Every model × city: MAE / RMSE / R². |
| `reports/ranked_metrics.csv` | The same, ranked best→worst within each city. |
| `reports/combined_scores.csv` | Both cities pooled into one score per model. |
| `reports/best_models.csv` | Winning model per city. |
| `reports/model_comparison_report.txt` | Human-readable comparison report. |
| `reports/run_summary.json` | Machine-readable run summary. |
| `predictions/<city>_<model>_holdout.csv` | Hold-out date/actual/predicted (16 files). |
| `predictions/submission.csv` | 416-week test forecast (best model per city). |

---

## STEP 7 — Visualization Catalogue (every figure that is generated)

> Each run writes **97 figures, all at 300 DPI**, in a single consistent style
> (fonts, palette, per-city colors: San Juan = blue, Iquitos = crimson). They
> fall into three tiers: (A) data-understanding, (B) per-model diagnostics, and
> (C) cross-model comparison.

### A. Data-understanding / EDA figures (generated once, before modeling)

**`outputs/figures/missing_values.png` — Missing-value diagnostics.**
Two panels. *Left:* a horizontal bar chart of the worst missingness per feature
(the max % missing across the San-Juan-train / Iquitos-train / test groups),
red for gapped, green for complete. *Right:* an annotated heatmap of % missing
per **feature × group**, features ordered worst-first, each cell labelled with
the exact percentage. **Reading it:** long/red bars are the gapped features
(e.g. `ndvi_ne`); the heatmap shows *which* group each gap lives in. **Justifies:**
that imputation is needed, and that the two cities miss *different* things —
San Juan lacks satellite NDVI, Iquitos lacks ground weather-station readings —
one reason each city is cleaned and modeled independently.

**`outputs/figures/target_overview.png` — Target behaviour & the log transform.**
One row per city, two columns. *Left:* weekly `total_cases` over time (the
epidemic curve, filled). *Right:* two overlaid histograms of the target, raw
(red) vs `log1p` (green), with the **skewness printed in the title**. **Reading
it:** the time series shows strong yearly seasonality and rare tall outbreak
spikes; the raw histogram is heavily right-skewed while the log version is
near-symmetric. **Justifies:** the single most important preprocessing choice —
training on `log1p(total_cases)` so models optimise evenly instead of being
dominated by a handful of spikes.

**`outputs/figures/<city>/eda/correlation_heatmap.png` — Correlation heatmap
(one per city: `sj`, `iq`).**
A lower-triangle Pearson-correlation heatmap of all base weather features **and**
the target (blue = negative, red = positive, white ≈ 0). **Reading it:** dark
clusters reveal groups of near-duplicate variables (the several temperature
columns are mutually collinear); the target's row shows which raw variables move
with case counts. **Justifies:** that many features are redundant/collinear
(motivating feature selection) and that temperature/humidity are most related to
cases.

**`outputs/figures/<city>/eda/feature_selection_mi.png` — Mutual-information
feature ranking (one per city).**
A horizontal bar chart of the top features by **mutual-information score** with
the target; the selected top-30 bars are green, unselected are grey. **Reading
it:** longer green bar = more predictive; you can read off exactly which
engineered feature (which lag / which rolling window) each model was allowed to
use. **Justifies:** the feature-selection step is fully auditable, and it shows
the two cities select **different** features — reinforcing per-city modeling.

### B. Per-model diagnostic figures (one set per model, per city)

Written to `outputs/figures/<city>/<model>/`. There are **8 models × 2 cities =
16 folders**. The four linear models get 5 figures each; the four tree models
additionally get a feature-importance plot (6 each).

**`actual_vs_predicted.png` — Actual vs Predicted line chart.** The hold-out
period over time: actual cases (navy solid, with markers) vs prediction (orange
dashed), with MAE / RMSE / R² in a corner box. **Reading it:** where the lines
hug, the model is right; the models track the seasonal shape but consistently
*under-shoot the tall peaks*. **Justifies:** the headline "does it work" view and
visual proof of the "weather glass ceiling".

**`residuals_over_time.png` — Residual plot.** Error (actual − predicted) week by
week, with a zero reference line and shaded area. **Reading it:** points near zero
are good; large positive spikes are the under-predicted outbreak weeks; clustered
(not random) errors reveal *structured* error. **Justifies:** shows *when* the
model fails, not just how much.

**`error_scatter.png` — Prediction-error scatter.** Predicted (y) vs actual (x)
with the ideal `y = x` diagonal; axes square/equal. **Reading it:** points on the
diagonal are perfect; points below the line at high actual values are
under-predictions. **Justifies:** the under-prediction as a *systematic bias*
across the range of case counts.

**`residual_hist.png` — Residual distribution histogram.** Histogram (+ KDE) of
all residuals, with the mean error (dashed) and zero (solid) marked. **Reading
it:** a tall symmetric bump on zero = unbiased errors; a long right tail = a few
big under-predictions. **Justifies:** the model is roughly unbiased on ordinary
weeks, and quantifies the "few large misses" pattern.

**`feature_importance.png` — Feature importance (tree models only).** A
horizontal bar chart of the top ~20 features by the model's internal importance.
**Reading it:** longest bars are the variables the model leaned on (typically
rolling means of humidity, dew-point, temperature). **Justifies:** interpretability
— the model learned biologically sensible drivers. (Linear models use coefficients
instead, so their dashboard panel says so explicitly.)

**`dashboard.png` — Combined 6-panel diagnostic dashboard.** A single
publication-ready page combining all of the above plus a text panel with the
model, city, MAE/RMSE/R² and the tuned hyperparameters. **Justifies:** the
one-image-per-model summary suitable for a slide or paper.

### C. Cross-model comparison figures

**`outputs/figures/<city>/all_models_timeline.png` — All-models overlay (one per
city).** The actual hold-out curve (bold black) with **every** model's prediction
overlaid in distinct colors, each labelled with its MAE. **Reading it:** all
models agree on the seasonal shape and under-shoot the peaks *together*.
**Justifies:** that the error ceiling is a *data* limit, not a bad-model problem.

**`outputs/figures/model_comparison.png` — Model comparison bar chart.** Three
panels (MAE, RMSE, R²), each a grouped bar chart of all 8 models with San Juan
and Iquitos as separate colored bars, sorted best-first. **Reading it:** shorter
MAE/RMSE = better, taller R² = better; San Juan's bars are ~2–3× taller because
it has more cases. **Justifies:** the final model-selection decision at a glance.

### Figure-count summary

| Group | Files |
| :--- | ---: |
| Global EDA (`missing_values`, `target_overview`, `model_comparison`) | 3 |
| Per-city EDA (`correlation_heatmap` + `feature_selection_mi`, × 2 cities) | 4 |
| Per-city all-models timeline (× 2 cities) | 2 |
| Per-model diagnostics (4 linear × 5 + 4 tree × 6, × 2 cities) | 88 |
| **Total** | **97** |

---

## Reproducibility & how to run

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py            # regenerates all outputs/
```

A single global seed (`project.random_seed = 42`) seeds Python, NumPy and every
estimator, so repeated runs are identical — including the stochastic MI estimator
and `RandomizedSearchCV`. All paths, splits, tuning grids and plot styling live
in `config/config.yaml`; the code contains no magic numbers.

> **Environment note.** LightGBM must be **≥ 4.6** to work with scikit-learn
> ≥ 1.6/1.9 (earlier LightGBM passes a removed `force_all_finite` kwarg and
> crashes). This is pinned in `requirements.txt`.

---

## STEP 8 — Autoregressive (nowcast) mode: reaching RMSE ≈ 10

**Why this exists.** The weather-only models above have RMSE ~27 (San Juan) and
~12.5 (Iquitos). Diagnostic experiments confirmed that (a) wider/deeper tuning of
XGBoost/Random Forest does **not** lower RMSE — they are already on the floor —
and (b) that floor is set by the rare outbreak **peaks**, which weather alone
cannot foresee (an *optimistic* "perfect-except-can't-see-outbreaks" bound is
RMSE ≈ 16 for San Juan — already above 10). So there is **no weather-only,
tuning-only path to RMSE ≈ 10.**

**The lever that works: past case counts.** Weekly dengue cases are strongly
autocorrelated, so adding **lagged `total_cases` features** (1–4 weeks + a 4-week
rolling mean) lets the models "ride" an outbreak. This is enabled with a single
config flag and is **OFF by default**:

```yaml
feature_engineering:
  autoregressive: true          # add past-case features (1-week-ahead nowcast)
```

Run it with `python scripts/run_autoregressive.py`, which writes to
`outputs_autoregressive/` (the default weather-only outputs under `outputs/` are
left untouched). Because future test weeks have no known case history, the test
set is forecast **recursively** — each prediction is fed back as the next week's
lag.

**Results — autoregressive hold-out (RMSE now ≈ 8–12).**

| City | Model | MAE | RMSE | R² |
| :--- | :--- | ---: | ---: | ---: |
| San Juan | LightGBM | 6.67 | **10.49** | 0.866 |
| San Juan | **XGBoost** | 6.71 | **11.23** | 0.846 |
| San Juan | **Random Forest** | 6.94 | **11.82** | 0.829 |
| San Juan | Decision Tree | 6.98 | 11.29 | 0.845 |
| Iquitos | Decision Tree | 5.31 | **8.20** | 0.518 |
| Iquitos | **Random Forest** | 5.40 | **8.77** | 0.449 |
| Iquitos | LightGBM | 5.71 | 8.93 | 0.429 |
| Iquitos | **XGBoost** | 5.86 | **9.29** | 0.381 |

Compared with weather-only, RMSE improves **~2.5×** and R² jumps from ~0 to
**0.83–0.87** (San Juan). The tuned Random Forest / XGBoost now land right around
the requested **RMSE ≈ 10** (San Juan 11.2–11.8; Iquitos 8.8–9.3).

### Two signals, combined — the head-to-head that proves it

The gain comes from feeding the models **two complementary families of signal at
once**:

* **Climate / weather** (rainfall, temperature, vegetation + lags/rolling/
  seasonality) — the long-term *environmental setup* for mosquito breeding
  ("is it a high-risk season?").
* **Recent case history** (lagged weekly `total_cases`) — the immediate
  *transmission reality* ("has the outbreak already started?").

`scripts/nowcast_demo.py` isolates exactly this effect: it trains the two
highlighted tree models on **climate only** vs **climate + cases** through the
identical leakage-safe pipeline, and writes the comparison and predictions to
`outputs/nowcast_demo/`.

| City | Model | Climate only (RMSE) | Climate + Cases (RMSE) | Error cut |
| :--- | :--- | ---: | ---: | ---: |
| San Juan | XGBoost | 27.76 | **11.23** | **−60%** |
| San Juan | Random Forest | 27.29 | **11.82** | −57% |
| Iquitos | Random Forest | 13.23 | **8.77** | −34% |
| Iquitos | XGBoost | 12.46 | **9.29** | −25% |

**Accuracy (R², share of week-to-week variation explained):**

| City | Climate only | Climate + Cases |
| :--- | ---: | ---: |
| San Juan | ~0.07 (≈7%) | **0.85 (≈85%)** |
| Iquitos | ~0.00 (≈0%) | **0.45 (≈45%)** |

**Net effect:** average hold-out RMSE falls **20.2 → 10.3 (~49% less error)**,
and San Juan's explained variance rises from a near-useless **~7% to ~85%**.
Combining the two signals bridges environmental *theory* and the actual
*outbreak* on the ground — neither signal alone gets close.

**The trade-off you must state honestly.** This is no longer a pure multi-week
weather *forecast* — it is a **1-week-ahead nowcast** that assumes recent case
counts are known at prediction time. That is realistic operationally (health
departments know last week's counts) but is exactly what the DengAI competition
disallows, and the recursively-generated long-horizon test submission degrades as
error compounds. **Use weather-only (`outputs/`) as the DengAI-legit forecast;
use autoregressive (`outputs_autoregressive/`) as the higher-accuracy operational
nowcast.** Both are produced and documented.

---

## Where the pipeline stands

| Phase | Status |
| :--- | :--- |
| Data loading & per-city separation | ✅ done — Step 1 |
| Data cleaning & missing-value imputation | ✅ done — Step 2 |
| Feature engineering (lags + rolling + seasonality) | ✅ done — Step 3 |
| Target transformation (`log1p`) | ✅ done — Step 4 |
| Feature selection (`SelectKBest`, mutual information) | ✅ done — Step 4 |
| Feature scaling (`StandardScaler`) | ✅ done — Step 4 |
| Temporal split + 8-model benchmark + tuning (tree models) | ✅ done — Step 5 |
| Evaluation (MAE / RMSE / R²) + ranking + submission | ✅ done — Step 6 |
| Visualizations (97 figures @ 300 DPI) | ✅ done — Step 7 |
| Autoregressive nowcast mode (opt-in) | ✅ done — Step 8 |
| Automated test suite (`pytest`) | ✅ done — 22 tests |

---

## Testing

A `pytest` suite under `tests/` guards the pipeline (run with `python -m pytest`):

* **Config & models** — config loads with absolute paths; exactly the 8 permitted
  models are exposed; only tree models are tunable.
* **Data & features** — raw row counts and chronological order; redundant column
  dropped; zero NaNs after cleaning and after feature engineering; all test weeks
  retained; seasonality/lag/rolling features present; autoregressive mode adds
  case features (real on train, NaN placeholders on test).
* **Leakage & transform** — every hold-out week is strictly *after* every train
  week; the scaler/selector are train-fit and column-aligned; `log1p`↔`expm1`
  round-trips and clips negative counts.
* **Evaluation** — metric values, best-first ranking per city, pooled scoring.
* **Integration** — an untuned model driven through the real split/select/scale/
  evaluate stack (finite metrics, non-negative predictions), plus both the Grid
  and Randomized search paths on tiny synthetic data.

All 22 tests pass in ~11 s (no full tuning sweep is triggered).

---

## Limitations & future work

1. **The weather glass ceiling.** Dengue outbreak **peaks are systematically
   under-predicted**: weather alone cannot explain the *magnitude* of the rarest
   large outbreaks, which are driven by unmeasured human/immunological factors.
   MAE is near the practical floor for weather-only inputs, and **R² is
   intrinsically low** (even slightly negative for some Iquitos models) because
   epidemic counts are spiky — **MAE, not R², is the meaningful metric here.**
   Breaking the ceiling needs external data, not more tuning (the autoregressive
   mode of Step 8 is the one lever that helps, at the cost of reframing the task).

2. **Single temporal hold-out, not rolling-origin CV.** Reported numbers come
   from one chronological 75/25 split. This is defensible for a short series and
   matches deployment, but a rolling-origin (walk-forward) evaluation would give
   tighter variance estimates. *Future work.*

3. **R-notebook fidelity.** The reference (rpubs.com/jencheng/dengue) is
   JavaScript-rendered and could not be scraped, so this pipeline reproduces the
   **standard DengAI workflow and analysis style**, not a verified line-by-line
   match — it is *inspired by*, not a literal translation of, the R notebook.

4. **Autoregressive submission is a nowcast, not a competition entry.** In AR
   mode the 416-week test set is forecast recursively (predictions feed forward
   as lags), so error compounds over the horizon; it is an operational nowcast,
   not a DengAI-valid multi-week forecast. The weather-only `outputs/` remains the
   competition-legitimate deliverable.
