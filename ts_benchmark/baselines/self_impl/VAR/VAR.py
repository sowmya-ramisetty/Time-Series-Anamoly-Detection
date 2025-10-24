from typing import Optional, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR

from ts_benchmark.models import ModelBase


class VAR_model(ModelBase):
    """
    VAR class.

    This class encapsulates a process of using VAR models for time series prediction.
    """

    def __init__(
        self,
    ):
        self.scaler = StandardScaler()
        self.model_args = {}

    @staticmethod
    def required_hyper_params() -> dict:
        """
        Return the hyperparameters required by VAR.

        :return: An empty dictionary indicating that VAR does not require additional hyperparameters.
        """
        return {}

    def forecast_fit(
        self,
        train_data: pd.DataFrame,
        *,
        covariates: Optional[Dict] = None,
        train_ratio_in_tv: float = 1.0,
        **kwargs
    ) -> "ModelBase":
        """
        Train the model.

        :param train_data: Time series data used for training.
        :param covariates: The covariates of time series.
        :param train_ratio_in_tv: Represents the splitting ratio of the training set validation set. If it is equal to 1, it means that the validation set is not partitioned.
        :return: The fitted model object.
        """

        self.scaler.fit(train_data.values)
        train_data_value = pd.DataFrame(
            self.scaler.transform(train_data.values),
            columns=train_data.columns,
            index=train_data.index,
        )
        model = VAR(train_data_value)
        self.results = model.fit(13)

    def forecast(
        self,
        horizon: int,
        testdata: pd.DataFrame,
        covariates: Optional[Dict] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Make predictions.

        :param horizon: The predicted length.
        :param testdata: Time series data used for prediction.
        :param covariates: The covariates of time series.
        :return: An array of predicted results.
        """
        train = pd.DataFrame(
            self.scaler.transform(testdata.values),
            columns=testdata.columns,
            index=testdata.index,
        )
        z = self.results.forecast(train.values, steps=horizon)

        predict = self.scaler.inverse_transform(z)
        return predict

    def __repr__(self) -> str:
        """
        Returns a string representation of the model name.
        """
        return self.model_name
