# Depositor

A lightweight e2e ML project that trains a model to predict whether a bank customer will subscribe to a term deposit. This project includes a complete training pipeline, FastAPI inference service, and Dockerized setup.


# Prerequisites
Make sure you have these installed:
- Python 3.11+
- pip
- make
- Docker and Docker Compose

If `make` or `Docker` aren’t installed, follow your OS-specific instructions:

```
sudo apt install make docker.io docker-compose -y
```

# How to run?

### Steps to run locally

Make use of the make commands to run the application. With the following commands

```
# Install dependencies
make install

# Train model and save artifacts
make train

# Run FastAPI server
make api
# → http://localhost:8000
```

The api has a GET `/health` and a POST `/predict` endpoints.


### Steps to run with Docker

1. Build and run trainer

`make build-trainer`

2. Build and run Api(serving/inference)

`make build-api`

3. Build and run both services

`make build-both`

4. Tear down all container

`make down`

That's it. Use the `/predict` end point to run predictions. Example payload schema

```
{
  "customers": [
    {
      "age": 35,
      "job": "blue-collar",
      "marital": "married",
      "education": "secondary",
      "default": "no",
      "balance": 1200.50,
      "housing": "yes",
      "loan": "no",
      "contact": "cellular",
      "day": 5,
      "month": "may",
      "duration": 180,
      "campaign": 2,
      "pdays": 999,
      "previous": 0,
      "poutcome": "unknown"
    }
  ]
}

```

You can use the cURL command 

```
curl -X POST "http://127.0.0.1:8000/predict?threshold=0.5" \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      {
        "age": 35,
        "job": "blue-collar",
        "marital": "married",
        "education": "secondary",
        "default": "no",
        "balance": 1200.50,
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day": 5,
        "month": "may",
        "duration": 180,
        "campaign": 2,
        "pdays": 999,
        "previous": 0,
        "poutcome": "unknown"
      }
    ]
  }'

```

Or simply use Postman or other API testing clients.


## Additional Questions

1. Testing the training code:
Unit-test on preprocessing (imputers/encoders produce expected shapes/values), feature lists (no missing/extra columns), label encoding, and data splits (stratification). Add integration tests that fit a tiny pipeline on a synthetic dataset: no exceptions, predict_proba works, and basic metric thresholds (>0 AUC). Include schema tests that validate the schema of the payload, etc, and serialization tests (joblib save/load).

2. Tracking experiments (configs, data, params, metrics):
Use a run tracker (MLflow) to log config (YAML hash), code version (git SHA), for data fingerprint ( DVC commit), hyperparameters, metrics, and artifacts (model, plots). Store raw data via DVC (or dataset versioning). save the exact requirements.txt/conda env/pyproject.toml to ensure result reproducibility

3. Fluctuating results without changes:
Likely nondeterminism: missing random seeds (train/test split, model, CV), parallel threads, or data order randomness; occasionally data leakage or environment drift (library versions). Fix by setting all seeds (numpy, PYTHONHASHSEED, sklearn, XGBoost random_state) by having a master config that applies the same seed to all downstream seed functions. Also ensuring the environment hasn't changed with inclusion/exclusion of packages that could lead to varied results

4. Retraining strategy:
Define triggers (time-based cadence(CRON), volume of new data, data/label drift/Model drift, or performance degradation in production). Use a champion–challenger workflow: automatically train candidates with the latest data, evaluate against fixed validation/test (and shadow live traffic if possible), promote only if metrics and bias/safety checks pass. Automate via CI/CD, version everything, and add rollback + monitoring on post-deploy performance.


5. Model Promotion in production
Staging & shadow —> deploy beside prod to score identical live traffic without affecting users; compare real-time metrics (AUC/PR, latency, error rate, drift).
Canary/A-B -> start with 1–5% traffic,  have auto-rollback errors, etc; expand gradually with more checks and with caution
Observability -> live dashboards for performance, drift, input schema, calibration; alerts with clear rollback procedure.
Governance -> versioned artifacts, approval checks, and migration plan. Only promote "challenger" to "champion" if it beats baseline on predefined KPIs and safety checks.

6. API contract change
-  Version the interface: expose /v1/predict (old schema) and add /v2/predict (new schema + preprocessing). Keep v1 stable and  make v2 the default only after adoption.
- Add a middlware/interface that maps v1 requests to v2. Validate the inputs, reformulate the results to match the expected response schema with proper exception handling.
- Have a log of which request was served by which version to see any glaring inconsistencies and enough confidence to move to the latest version
- Rollout safely: shadow or canary the v2 model behind the new endpoint; compare v1 vs v2 metrics, latency, and performance, ensure there is instant rollback capability.
- Customer comms & migration: Provide all the information needed to customers about the change with examples, documentation, etc by following proper communication channels.


7. API observability

Practical metrics to measure on API
- API health -> throughput, latency, timeouts, response time, etc
- input data quality -> schema validation, missing attributes, payload size
- data drift/model drift -> score mean/std shift, feature drift (PSI/KL) per top features, class/acceptance rate drift, share of traffic per model version.

when to alert
- Availabilty - Alert if the API is too slow or often fails. For example, if more than 1% of requests fail with a 5xx error for 5 minutes, or if the average slowest (p95) requests take over 500 ms for 5–10 minutes.
- Correctness: Alert if the data coming in looks wrong.
- Drift or behavior changes: Alert if the model’s behavior or input data drifts, e.g. feature distributions change a lot, or the acceptance rate or key metric moves way outside normal range (more than 3 standard deviations) for a while.
- Deployment issues: Alert if the model or preprocessing fails to load, there’s a version mismatch, or a new "shadow" model behaves very differently from the production one.
