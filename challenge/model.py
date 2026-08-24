import logging
from pathlib import Path
from typing import List, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

logger = logging.getLogger(__name__)


class InputDataException(Exception):
    """Exception class for errors in input data for model"""


class ModelNotLoadedException(Exception):
    """Exception class for predictions requested without a trained model"""


class DelayModel:
    ROOT_PATH = Path(__file__).resolve().parents[1]
    MODEL_PATH = ROOT_PATH.joinpath('challenge', 'model.joblib')
    METADATA_PATH = ROOT_PATH.joinpath('challenge', 'model_metadata.json')

    DELAY_THRESHOLD_MINUTES = 15
    DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'
    SCHEDULED_TIME = 'Fecha-I'
    REAL_TIME = 'Fecha-O'
    
    PRE_FEATURE_COLS = ['OPERA', 'TIPOVUELO', 'MES']
    FEATURE_COLS = [
        "OPERA_Latin American Wings",
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air"
    ]

    # Categories seen in the training data, used for validation on api.
    MIN_MONTH = 1
    MAX_MONTH = 12
    TIPOVUELO_CATEGORIES = ['I', 'N']
    OPERA_CATEGORIES = [
        "Aerolineas Argentinas",
        "Aeromexico",
        "Air Canada",
        "Air France",
        "Alitalia",
        "American Airlines",
        "Austral",
        "Avianca",
        "British Airways",
        "Copa Air",
        "Delta Air",
        "Gol Trans",
        "Grupo LATAM",
        "Iberia",
        "JetSmart SPA",
        "K.L.M.",
        "Lacsa",
        "Latin American Wings",
        "Oceanair Linhas Aereas",
        "Plus Ultra Lineas Aereas",
        "Qantas Airways",
        "Sky Airline",
        "United Airlines"
    ]

    RANDOM_STATE = 42


    def __init__(
        self
    ):
        self._model = None

        if self.MODEL_PATH.exists():
            self._model = joblib.load(self.MODEL_PATH)
            logger.info(f"Model artifact loaded from {self.MODEL_PATH}")
        else:
            logger.warning(
                f"No model artifact at {self.MODEL_PATH}. Call fit() before predict()."
            )

    def _check_required_columns(self, data: pd.DataFrame, target_column: str) -> None:
        required_columns = self.PRE_FEATURE_COLS.copy()

        if target_column:
            required_columns += [self.SCHEDULED_TIME, self.REAL_TIME]

        missing_columns = sorted(set(required_columns) - set(data.columns))

        if missing_columns:
            message = f"Missing required columns: {', '.join(missing_columns)}"
            logger.warning(message)
            raise InputDataException(message)


    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        # Check data validity
        self._check_required_columns(data, target_column)

        # One hot encoding, then keep the 10 most important features (FEATURE_COLS).
        # Categories missing from 'data' are filled with zeros, so the output
        # columns are always FEATURE_COLS, in that order, for any input.
        features = pd.get_dummies(
            data[self.PRE_FEATURE_COLS],
            columns=self.PRE_FEATURE_COLS
        )
        features = features.reindex(columns=self.FEATURE_COLS, fill_value=0).astype(int)

        if not target_column:

            return features
        else:
            # Calculate delay and then target
            real_time = pd.to_datetime(data[self.REAL_TIME], format=self.DATETIME_FORMAT)
            scheduled_time = pd.to_datetime(data[self.SCHEDULED_TIME], format=self.DATETIME_FORMAT)
            time_diff_minutes = ((real_time - scheduled_time).dt.total_seconds())/60
            target = np.where(time_diff_minutes > self.DELAY_THRESHOLD_MINUTES, 1, 0)
            target = pd.DataFrame(data={target_column: target}, index=features.index)

            return features, target


    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        self._model = LogisticRegression(class_weight='balanced')

        x_train, y_train = shuffle(features, target, random_state=self.RANDOM_STATE)
        self._model.fit(x_train, y_train.values.ravel())

        logger.info(f"Model fitted on {len(x_train)} rows")

        return

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            message = "No model loaded, call fit() first."
            logger.warning(message)
            raise ModelNotLoadedException(message)

        y_pred = self._model.predict(features)
        y_pred = y_pred.astype(int).tolist()

        return y_pred
