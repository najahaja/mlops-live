# ML-Ops-Live — Interview Quick Reference & Cheat Sheet

> **Project Name:** ML-Ops-Live — NYC Taxi Trip-Duration Prediction & MLOps Platform  
> **Target Roles:** AI Engineer, Machine Learning Engineer, MLOps Engineer, Applied AI Engineer, Backend Engineer, Software Engineer  
> **Market Target:** UAE & Saudi Arabia Technology Ecosystems (Enterprise, Tech Scale-ups, Government AI Initiatives)

---

## 1. Verified Numerical Reference Sheet (The "Ground Truth" Numbers)

Memorize these exact figures before any interview. Every metric below is backed by a verified artifact in the repository.

### Dataset & Partitions
| Metric / Parameter | Value | Source / Evidence |
| :--- | :--- | :--- |
| **Data Source** | NYC TLC Green Taxi Monthly Trip Records (`green_tripdata_YYYY-MM.parquet`) | `src/mlops_live/data/ingest.py` |
| **Total Cleaned Records** | **102,189 trips** (Jan 2024: 52,529 + Feb 2024: 49,660) | `clean.py`, `feb_2024_out_of_time_evaluation.json` |
| **Jan 2024 Raw $\rightarrow$ Cleaned** | 56,551 $\rightarrow$ **52,529 rows** (92.89% retention) | `prepare_data.py`, `README.md` |
| **Feb 2024 Raw $\rightarrow$ Cleaned** | 53,577 $\rightarrow$ **49,660 rows** (92.69% retention) | `feb_2024_out_of_time_evaluation.json` |
| **Cleaning Filters** | `1.0 <= duration_minutes <= 60.0`, `0.01 <= trip_distance <= 100.0`, `PULocationID > 0`, `DOLocationID > 0` | `src/mlops_live/data/clean.py` |
| **Train Split (70%)** | **36,770 rows** (`2023-12-31 14:38:47` to `2024-01-23 07:46:22`) | `data/processed/train.parquet` |
| **Validation Split (15%)**| **7,879 rows** (`2024-01-23 07:46:37` to `2024-01-27 13:37:47`) | `data/processed/val.parquet` |
| **Holdout Test Split (15%)**| **7,880 rows** (`2024-01-27 13:40:09` to `2024-01-31 23:57:29`) | `data/processed/test.parquet` |
| **Chronological Split Method**| Stable mergesort (`df.sort_values(by="lpep_pickup_datetime", kind="mergesort")`) | `src/mlops_live/data/split.py` |
| **Feature Contract (6)** | `PULocationID`, `DOLocationID`, `trip_distance`, `passenger_count`, `pickup_hour`, `pickup_day_of_week` | `FEATURE_COLUMNS` in `clean.py` |
| **Target Variable** | `duration_minutes` = `(dropoff_dt - pickup_dt).dt.total_seconds() / 60.0` | `compute_trip_duration()` in `clean.py` |

### Model Benchmark & Performance
| Evaluation Split | Model | Alias / Version | MAE (min) | RMSE (min) | $R^2$ Score | Error P50 / P90 / P95 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Validation (Jan)** | `dummy_median` | — | 6.641 | 9.780 | -0.098 | — |
| **Validation (Jan)** | `ridge_baseline` | `@challenger` (`v1`) | 3.416 | 5.206 | 0.689 | — |
| **Validation (Jan)** | **`hist_gbdt_candidate`** | **`@champion` (`v2`)** | **2.918** | **4.722** | **0.744** | — |
| **Holdout Test (Jan)**| `ridge_baseline` | `@challenger` (`v1`) | 3.200 | 4.810 | 0.704 | 2.22 / 6.93 / 9.64 min |
| **Holdout Test (Jan)**| **`hist_gbdt_candidate`** | **`@champion` (`v2`)** | **2.750** | **4.352** | **0.758** | **1.68 / 6.28 / 9.00 min** |
| **Out-of-Time (Feb)** | `ridge_baseline` | `@challenger` (`v1`) | 3.259 | 4.876 | 0.693 | 2.30 / 6.98 / 9.72 min |
| **Out-of-Time (Feb)** | **`hist_gbdt_candidate`** | **`@champion` (`v2`)** | **2.773** | **4.357** | **0.755** | **1.71 / 6.30 / 8.98 min** |

- **Holdout Gain (Jan 2024):** Candidate cuts MAE by **-14.06%** (`-0.450 min`), RMSE by **-9.51%** (`-0.457 min`), $R^2$ gain **+0.054**.
- **Out-of-Time Gain (Feb 2024):** Candidate cuts MAE by **-14.92%** (`-0.486 min`), RMSE by **-10.64%** (`-0.519 min`), $R^2$ gain **+0.062**.

### Serving Latency & Load Testing (HTTP Uvicorn localhost)
| Metric | Value | Reference Artifact |
| :--- | :--- | :--- |
| **Methodology** | `local_uvicorn_localhost_tcp_http` (TCP HTTP over loopback) | `reports/generated/http_load_test_report.json` |
| **Server Target** | `http://127.0.0.1:8010/predict` | `scripts/load_test.py` (`--spawn-server`) |
| **Concurrency / Requests** | 4 worker threads / 50 total POST requests | `http_load_test_report.json` |
| **Error Rate** | **0.0%** (50/50 HTTP 200 OK) | `http_load_test_report.json` |
| **Throughput** | **24.72 requests/second** | `http_load_test_report.json` |
| **Server Inference Latency**| P50: **112.67 ms** \| P95: **255.01 ms** \| P99: **262.29 ms** | `http_load_test_report.json` |
| **Client HTTP Round-Trip** | P50: **145.99 ms** \| P95: **292.38 ms** \| P99: **323.96 ms** | `http_load_test_report.json` |
| **SLA Gate (`P95 <= 500ms`)**| **PASSED (`true`)** | `http_load_test_report.json` |

### Drift Detection & Controlled Retraining Sandbox
| Step | Action & Dataset | Verified Metric / Outcome |
| :--- | :--- | :--- |
| **Negative Control** | Jan Train (`36,770`) vs. Jan Val (`7,879`) | `overall_drift_detected = False`; `pickup_day_of_week` flagged as `insufficient_data` (missing Sun/Mon in 4.2-day slice); `--only-if-drifted` skips retraining. |
| **Positive Control** | Short-haul baseline (`trip_dist <= 3.0 mi`, `26,902` rows) vs. Shifted Val (`trip_dist > 3.5 mi`, `1,820` rows) | `overall_drift_detected = True`; `trip_distance` PSI = **8.268**, KS = **1.000**; prediction PSI = **6.264**, KS = **0.780**. |
| **Retraining Gate** | Retrained `hist_gbdt` (`v4`) vs. Incumbent `hist_gbdt` (`v2`) on shifted data | Incumbent MAE: **9.040 min** $\rightarrow$ Challenger MAE: **5.554 min** (**38.56% reduction**); Gate **PASSED**. Staged as `@challenger` `v4`. |
| **Human Governance** | Human promotion $\rightarrow$ verification $\rightarrow$ instant rollback | `v4` promoted to `@champion` by actor `najahaja`; then rolled back to `v2`. 4 audit events logged in SQLite. |

### Quality & Engineering Verification
- **Automated Test Count:** **79 passing tests** (in ~110 seconds via `pytest -q`).
- **Code Linter / Formatter:** `ruff check .` (0 errors), `ruff format --check .` (46 files formatted).
- **Environment:** Python 3.11.9 on Windows 11, pinned via `requirements.txt` / `requirements-dev.txt` / `requirements-lock.txt`.
- **Streamlit Web UI:** `streamlit_app.py` (root delegator) & `webapp/streamlit_app.py` with 5 interactive tabs.

---

## 2. Platform Architecture Summary

```
                      +---------------------------------------+
                      |       NYC TLC Green Taxi Parquet      |
                      |    (Monthly CloudFront Public Feed)   |
                      +-------------------+-------------------+
                                          |
                                          v
                      +-------------------+-------------------+
                      |      Data Ingestion & Cleaning        |
                      |   clean.py + mergesort split.py       |
                      +-------------------+-------------------+
                                          |
                +-------------------------+-------------------------+
                |                         |                         |
                v                         v                         v
        train.parquet (70%)        val.parquet (15%)        test.parquet (15%)
          (36,770 rows)              (7,879 rows)              (7,880 rows)
                |                         |                         |
                v                         v                         |
        +-------+-------------------------+-------+                 |
        |        Model Training & Registry        |                 |
        |  Dummy / Ridge / HistGBDT Regressor     |                 |
        |  MLflow Tracking (sqlite:///mlflow.db)  |                 |
        |  @champion (v2) & @challenger (v1)      |                 |
        +-------------------+---------------------+                 |
                            |                                       |
                            v                                       |
        +-------------------+---------------------+                 |
        |     Production Serving & Feedback       |                 |
        |  FastAPI (scripts/serve.py)             |                 |
        |  POST /predict  |  POST /feedback       |                 |
        |  GET /dashboard |  GET /api/dashboard   |                 |
        +-------------------+---------------------+                 |
                            |                                       |
                            v                                       |
        +-------------------+---------------------+                 |
        |     SQLite Operational Store            |                 |
        |  predictions.db: prediction_logs,       |                 |
        |  prediction_feedback, drift_reports,    |                 |
        |  model_lifecycle_audit                  |                 |
        +-------------------+---------------------+                 |
                            |                                       |
                            v                                       |
        +-------------------+---------------------+                 |
        |     Drift Detection & Governance        |                 |
        |  check_drift.py (PSI + KS test)         |                 |
        |  retrain_and_evaluate.py (--only-if...) |                 |
        |  manage_model.py (--approve / --rollback|                 |
        +-------------------+---------------------+                 |
                            |                                       v
                            |                         +-------------+-------------+
                            |                         | Isolated Holdout Eval     |
                            |                         | scripts/evaluate_holdout  |
                            |                         +---------------------------+
                            v
        +-------------------+---------------------+
        |     Interactive Streamlit UI            |
        |  streamlit_app.py & webapp/             |
        |  Read-only demo for recruiters/users    |
        +-----------------------------------------+
```

---

## 3. The 20 High-Priority Questions to Master First

1. **What is ML-Ops-Live, and what business problem does it solve?** (Section A, Q1 & Q2)
2. **Walk me through the pipeline from raw NYC TLC data to a served prediction.** (Section A, Q4)
3. **Why did you use chronological splitting instead of random train_test_split?** (Section B, Q8)
4. **How did you prevent data leakage in your feature pipeline?** (Section B, Q5)
5. **Why did HistGradientBoostingRegressor outperform Ridge Regression by 14.1%?** (Section C, Q5)
6. **Why is MAE the primary optimization metric rather than RMSE or $R^2$?** (Section C, Q7)
7. **What is out-of-time evaluation, and what did your February 2024 test prove?** (Section C, Q11)
8. **How does your MLflow setup manage model versions, and what are `@champion` and `@challenger` aliases?** (Section D, Q6)
9. **What happens during inference if MLflow or the database is down?** (Section D, Q8)
10. **Explain how FastAPI processes `POST /predict` and logs to SQLite without blocking.** (Section E, Q3)
11. **How does delayed ground-truth feedback work, and why can’t you measure live MAE immediately?** (Section F, Q4)
12. **How do you calculate Population Stability Index (PSI), and what do the thresholds mean?** (Section F, Q9)
13. **Why did you implement a 7-day calendar coverage safeguard in drift monitoring?** (Section F, Q12)
14. **How does the `--only-if-drifted` flag prevent unnecessary model retraining?** (Section G, Q2)
15. **Explain the champion/challenger evaluation gate and why you check both MAE and RMSE.** (Section G, Q4)
16. **How does transactional rollback work, and how do you protect production model artifacts?** (Section G, Q7)
17. **What is the difference between your in-process load test and real HTTP Uvicorn load test?** (Section H, Q8)
18. **Why can average latency be dangerous in MLOps, and what were your P50 and P95 numbers?** (Section H, Q10)
19. **What is the weakest part of your architecture today, and what would break first under 1,000 req/s?** (Section K, Q10)
20. **If you had another month and a cloud budget, how would you evolve this to enterprise scale?** (Section M, Q1)
