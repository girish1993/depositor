import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Tuple

import joblib
import pandas as pd
from pydantic.dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.data.loader import Loader
from src.preprocessing.preprocessor import Preprocessor


@dataclass
class ModelMetrics:
    accuracy: float
    roc_auc: float
    f1: float
    pr_auc: float


class Trainer:
    MODEL_MAP = {
        "xgb": XGBClassifier,
        "lr": LogisticRegression,
    }

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.model = None
        self.pipeline = None
        self.label_encoder = LabelEncoder()
        self.metrics = {}

    def _load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        data_config = self.cfg.get("data")
        csv_config = self.cfg.get("csv", {})

        df = Loader.read_csv(file_path=data_config.get("file_path"), csv_cfg=csv_config)
        target_col = data_config.get("target")

        # Encode the target

        y = self.label_encoder.fit_transform(df[target_col])
        X = df.drop(columns=[target_col])

        return X, y

    def _split_data(
        self, X: pd.DataFrame, y: pd.Series, cfg: Dict
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        return train_test_split(
            X,
            y,
            test_size=cfg.get("test_size"),
            random_state=cfg.get("random_state"),
            stratify=y,
        )

    def _formulate_metrics(
        self, y_test: pd.Series, y_pred: pd.Series, y_prob: pd.Series
    ) -> Dict[str, float]:
        return asdict(
            ModelMetrics(
                accuracy=accuracy_score(y_true=y_test, y_pred=y_pred),
                roc_auc=roc_auc_score(y_true=y_test, y_score=y_prob),
                f1=f1_score(y_true=y_test, y_pred=y_pred),
                pr_auc=average_precision_score(y_true=y_test, y_score=y_prob),
            )
        )

    def _train(self):
        # Load data
        X, y = self._load_data()

        # prepare data
        preprocessor_pipeline = Preprocessor(
            categorical_feats=self.cfg["features"].get("categorical"),
            numerical_feats=self.cfg["features"].get("numerical"),
        ).build()

        # Instantiate traninig model
        self.model = self.MODEL_MAP.get(self.cfg["model"].get("type"), "xgb")(
            **self.cfg["model"].get("params")
        )

        # update the pipeline

        self.pipeline = Pipeline(
            [("preprocessor", preprocessor_pipeline), ("model", self.model)]
        )

        # split the dataset
        X_train, X_test, y_train, y_test = self._split_data(
            X=X, y=y, cfg=self.cfg.get("split")
        )

        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)
        y_prob = self.pipeline.predict_proba(X_test)[:1]

        self.metrics = self._formulate_metrics(
            y_test=y_test, y_pred=y_pred, y_prob=y_prob
        )

    def _save_artifacts(self):
        ts = datetime.strftime("%Y_%m_%d_%H_%M_%S")
        art_dir = self.cfg["artifacts"]["dir"]
        model_dir = os.path.join(art_dir, self.cfg["artifacts"]["model_subdir"])
        metrics_dir = os.path.join(art_dir, self.cfg["artifacts"]["metrics_subdir"])
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)

        joblib.dump(self.pipeline, os.path.join(model_dir, f"model_{ts}.joblib"))
        joblib.dump(
            self.label_encoder, os.path.join(model_dir, f"label_encoder_{ts}.joblib")
        )
        json.dump(
            self.metrics,
            open(os.path.join(metrics_dir, f"metrics_{ts}.json"), "w"),
            indent=2,
        )

    def run(self):
        print("Starting training..")
        self._train()
        self._save_artifacts()
        print("Training completed.")
