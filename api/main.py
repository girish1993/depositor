import logging
import os
from contextlib import asynccontextmanager

import joblib
import pandas as pd
import yaml
from data_models.schema_models import ApiRequest, ApiResponse
from fastapi import FastAPI, Query

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("api")
logger.setLevel(logging.INFO)

MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/models/latest_model.joblib")
LABEL_ENCODER_PATH = os.environ.get(
    "LABEL_ENCODER_PATH", "artifacts/models/latest_label_encoder.joblib"
)
CONFIG_PATH = os.environ.get("CONFIG_PATH", "configs/train.yaml")


def _load_model(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    app.state.model = joblib.load(MODEL_PATH)


def _load_label_encoder(app: FastAPI):
    if not os.path.exists(LABEL_ENCODER_PATH):
        raise FileNotFoundError(f"Model file not found at {LABEL_ENCODER_PATH}")
    app.state.label_encoder = joblib.load(LABEL_ENCODER_PATH)


def _load_config(app: FastAPI):
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)
        app.state.cfg = cfg
        app.state.expected_numeric = cfg["features"]["numerical"]
        app.state.expected_categorical = cfg["features"]["categorical"]
        app.state.expected_columns = (
            app.state.expected_numeric + app.state.expected_categorical
        )
    except FileNotFoundError as e:
        raise (f"Encountered an error while loading config {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Loading Model...")
        _load_model(app)

        logger.info("Loading Label encoder...")
        _load_label_encoder(app)

        logger.info("Loading config")
        _load_config(app)

        logger.info("Startup complete..")
        yield

        logger.info("Shutdown complete..")
    except Exception as e:
        logger.error(f"Encountered exception during startup {e}")


app = FastAPI(title="Deposit Predictor API", version="1.0")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": app.state.model}


@app.post("/predict", response_model=ApiResponse)
def predict(req: ApiRequest, threshold: float = Query(0.5, ge=0.0, le=1.0)):
    rows = [customer.model_dump() for customer in req.customers]
    df = pd.DataFrame(rows)

    expected_cols = app.state.expected_columns

    # handle if any missing cols and align it train data
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None

    df = df[expected_cols]

    probabilities = app.state.model.predict_proba(df)[:, 1]
    predictions = (probabilities >= threshold).astype(int).tolist()
    labels = app.state.label_encoder.inverse_transform(predictions).tolist()

    return ApiResponse(
        probabilities=probabilities, predictions=predictions, labels=labels
    )
