# ML-Ops-Live — Interview Unknowns, Limitations & Discrepancies

> **Purpose:** Senior interviewers and technical hiring managers probe beyond the happy path. This document details architectural constraints, trade-offs, historical discrepancies, and points you must be prepared to address honestly during interviews.

---

## 1. Verification Status Matrix of Engineering Claims

| Topic / Claim | Verified in Repository? | Status / Caveat to Acknowledge in Interview |
| :--- | :---: | :--- |
| **Model Metrics (MAE 2.75 min, $R^2$ 0.758)** | **Yes** | Fully verified on January 2024 holdout test (`7,880` rows) in `holdout_test_metrics.json`. |
| **Out-of-Time Generalization (Feb 2024)** | **Yes** | Verified on `49,660` cleaned rows in `feb_2024_out_of_time_evaluation.json`. |
| **FastAPI Serving & SQLite Logging** | **Yes** | Verified via unit tests (`test_serving.py`) and manual verification. |
| **HTTP Load Testing (24.7 req/s, P95 292 ms)** | **Yes** | Verified on local Windows machine via loopback TCP in `http_load_test_report.json`. |
| **Drift-Triggered Retraining & Governance** | **Yes** | Verified in an isolated sandbox in `controlled_drift_retraining_report.json`. |
| **Automated Test Suite (79 passed)** | **Yes** | Verified on Windows 11 (`pytest -q` exited code 0). |
| **Cloud Deployment on Streamlit Community Cloud** | **Prepared, Pending Live Deployment** | Code, entrypoints (`streamlit_app.py`), and dependencies (`webapp/requirements.txt`) are prepared and pushed to GitHub; live URL activation requires logging into `share.streamlit.io`. |
| **Docker / Kubernetes Deployment** | **No** | **Do not claim containerization.** The project was intentionally developed local-first on Windows without Docker or Kubernetes. |
| **Production Cloud Database (PostgreSQL)** | **No** | Local SQLite (`mlflow.db`, `predictions.db`) is used. |
| **Live Production Traffic / Real Users** | **No** | Predictions are generated via test clients and local benchmarks, not public consumer traffic. |

---

## 2. Discrepancies & Historical Evolution (Be Ready to Explain)

### Discrepancy 1: Test Count Evolution (54 $\rightarrow$ 55 $\rightarrow$ 76 $\rightarrow$ 78 $\rightarrow$ 79)
- **What happened:**
  - In initial Phase 7, the test suite had **54 tests**.
  - Adding the delayed ground-truth feedback test in `tests/test_serving.py` brought it to **55 tests**.
  - Adding the Streamlit webapp and sanitized exporter tests in `tests/test_webapp.py` added 21 tests (**76 tests**).
  - Adding Unicode mojibake and exact title assertions added 2 tests (**78 tests**).
  - Adding the root delegator `streamlit_app.py` test brought the final verified suite to **79 tests**.
- **Interview Response:** *"As we hardened the system—adding delayed ground-truth tracking, asset sanitization, and automated UI regression testing—we expanded test coverage from 54 to 79 tests."*

### Discrepancy 2: Negative Drift Control on 4.2-Day Validation Slice
- **What happened:** Initially, comparing Jan Train (23 days) to Jan Val (4.2 days) showed raw PSI of `2.03` on `pickup_day_of_week`. This looked like severe drift, but was actually an artifact of an incomplete calendar window (the validation slice spanned Tuesday–Saturday and had zero Sunday/Monday trips).
- **How it was fixed:** In `src/mlops_live/monitoring/drift.py`, `check_weekday_coverage()` was implemented. It checks if $\ge 7$ elapsed days are present and all weekdays have $\ge 5$ observations. If not, it sets `evaluation_status = "insufficient_data"` and prevents false retraining.
- **Interview Response:** *"This was a critical real-world bug we discovered. Short time windows produce artificial categorical drift. Implementing calendar-coverage validation prevented our automated pipeline from triggering false retraining."*

### Discrepancy 3: Controlled Drift Experiment vs. Live Traffic Drift
- **What happened:** Real February 2024 data showed natural stability (zero drift, PSI < 0.02). To prove that retraining, champion/challenger gating, and rollback actually work when drift occurs, we constructed a controlled sandbox experiment (`reports/generated/controlled_drift_retraining_report.json`) by filtering for long-haul trips (> 3.5 miles).
- **Interview Response:** *"I clearly distinguish natural multi-month stability from our controlled sandbox experiment. In real February data, there was no drift. So to validate that our drift gates and automated rollback work under stress, we engineered a synthetic regime shift in an isolated sandbox."*

---

## 3. Known Architectural Limitations & Trade-Offs

Be honest about these when asked *"What is the weakest part of your architecture?"*:

1. **SQLite Concurrency Bottleneck (Database Lock):**
   - SQLite uses file-level locking for writes. Under high write concurrency (multiple parallel `/predict` or `/feedback` requests), SQLite will throw `database is locked` (`OperationalError`).
   - *Fix in production:* Migrate to connection-pooled **PostgreSQL** or ingest logs asynchronously via a message queue (Kafka / RabbitMQ / AWS SQS).

2. **In-Memory Target Encoding on Location IDs:**
   - In `src/mlops_live/models/train.py`, `TargetEncoder` is used for `PULocationID` and `DOLocationID` with 5-fold cross-validation. If an entirely unseen taxi zone ID arrives at inference time, it falls back to the global target mean.
   - *Fix in production:* Maintain an explicit unknown-category smoothing strategy or pre-computed zone embeddings.

3. **In-Process Prediction Logging (Latency Impact):**
   - In `src/mlops_live/serving/app.py`, `log_prediction_record()` runs synchronously inside `predict()` before returning the response. This adds disk I/O to the client's request-response latency.
   - *Fix in production:* Use FastAPI `BackgroundTasks` or publish the prediction event to a background Redis queue/worker so inference latency is decoupled from database logging.

4. **Single-Node MLflow Registry:**
   - MLflow is running on a local SQLite database (`sqlite:///mlflow.db`) with artifacts on the local filesystem (`mlruns/`). If the server disk crashes, model lineage is lost.
   - *Fix in production:* Deploy a remote MLflow Tracking Server backed by Amazon RDS (PostgreSQL) and Amazon S3 or Google Cloud Storage for immutable model artifact storage.

5. **No Concept Drift Detection with Delayed Labels:**
   - PSI and KS tests detect **data drift** (covariate shift in inputs and output distributions), but cannot detect **concept drift** (where $P(Y|X)$ changes but $P(X)$ looks identical) until actual labels arrive via `/feedback`.

---

## 4. Questions to Investigate Before Specialist Interviews

If interviewing for specialized Senior/Lead roles, research these ahead of time:
- [ ] How would you configure **PostgreSQL connection pooling** (PgBouncer, SQLAlchemy `QueuePool`) for high-concurrency FastAPI logging?
- [ ] How would you package this application with **Docker multi-stage builds** and run it on **Kubernetes (EKS/GKE)** with Horizontal Pod Autoscaling (HPA)?
- [ ] How would you implement **Evidently AI** or **Great Expectations** for automated data contracts instead of custom PSI/KS functions?
- [ ] How would you handle **streaming inference** if ride-request predictions were consumed via Apache Kafka?
