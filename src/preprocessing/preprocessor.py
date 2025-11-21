from typing import List, Optional

from pydantic.dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Config:
    arbitrary_types_allowed = True


@dataclass(config=Config)
class Preprocessor:
    categorical_feats: List[str]
    numerical_feats: List[str]
    pipeline: Optional[Pipeline] = None

    def build(self) -> Pipeline:
        numeric_preprocessor = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )

        categorical_preprocessor = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]
        )

        self.pipeline = ColumnTransformer(
            [
                (
                    "numerical_transformations",
                    numeric_preprocessor,
                    self.numerical_feats,
                ),
                (
                    "categorical_transformations",
                    categorical_preprocessor,
                    self.categorical_feats,
                ),
            ]
        )

        return self.pipeline
