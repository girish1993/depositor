import logging
from typing import Any, Dict

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Loader:
    @staticmethod
    def read_yaml(file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, "r") as f:
                return yaml.safe_load(f.read())
        except FileNotFoundError as e:
            logger.error(f"File not found at {file_path} error: {e}")

    @staticmethod
    def read_csv(file_path: str, csv_cfg: Dict) -> pd.DataFrame:
        return pd.read_csv(
            file_path,
            sep=csv_cfg.get("separator", ","),
            quotechar=csv_cfg.get("quotechar", '"'),
        )
