"""Train the model and write its artifact and metadata.
Run with:  python -m challenge.train
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from challenge.logging_config import setup_logging
from challenge.model import DelayModel

logger = logging.getLogger(__name__)

TARGET_COLUMN = 'delay'
DATA_REL_PATH = Path('data', 'data.csv')
DATA_PATH = DelayModel.ROOT_PATH.joinpath(DATA_REL_PATH)


def _file_sha256(path: Path) -> str:
    """Compute the sha256 of a file."""
    with open(path, 'rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def _build_metadata(model: DelayModel, data: pd.DataFrame) -> dict:
    """Collect the metadata of the artifact being written."""
    return {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'data': {
            'path': DATA_REL_PATH.as_posix(),
            'sha256': _file_sha256(DATA_PATH),
            'rows': int(len(data))
        },
        'model': {
            'estimator': type(model._model).__name__,
            'params': model._model.get_params(),
            'features': DelayModel.FEATURE_COLS,
            'target': TARGET_COLUMN
        }
    }


def main() -> None:
    setup_logging()

    logger.info(f"Reading {DATA_PATH}")
    data = pd.read_csv(DATA_PATH, low_memory=False)

    model = DelayModel()
    features, target = model.preprocess(data=data, target_column=TARGET_COLUMN)
    model.fit(features=features, target=target)

    joblib.dump(model._model, DelayModel.MODEL_PATH)
    logger.info(f"Wrote artifact to {DelayModel.MODEL_PATH}")

    metadata = _build_metadata(model, data)
    DelayModel.METADATA_PATH.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    logger.info(f"Wrote metadata to {DelayModel.METADATA_PATH}")


if __name__ == '__main__':
    main()
