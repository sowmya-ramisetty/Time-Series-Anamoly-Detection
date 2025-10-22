# -*- coding: utf-8 -*-
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, NoReturn, List

import pandas as pd

from ts_benchmark.common.constant import (
    FORECASTING_DATASET_PATH,
    ANOMALY_DETECT_DATASET_PATH,
    ST_FORECASTING_DATASET_PATH,
)
from ts_benchmark.data.dataset import Dataset
from ts_benchmark.data.utils import load_series_info, read_data, read_covariates

logger = logging.getLogger(__name__)


class DataSource:
    DATASET_CLASS = Dataset

    def __init__(
        self,
        data_dict: Optional[Dict[str, pd.DataFrame]] = None,
        covariate_dict: Optional[Dict[str, Dict]] = None,
        metadata: Optional[pd.DataFrame] = None,
    ):
        self._dataset = self.DATASET_CLASS()
        self._dataset.set_data(data_dict, covariate_dict, metadata)

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    def load_series_list(self, series_list: List[str]) -> NoReturn:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support loading series at runtime."
        )


class LocalDataSource(DataSource):
    _INDEX_COL = "file_name"
    _COVARIATES_FOLDER_NAME = "covariates"

    def __init__(self, local_dataset_path: str, metadata_file_name: str):
        """
        Only metadata is loaded at init; data loaded on demand
        """
        self.local_data_path = local_dataset_path  # no extra folders
        self.local_covariates_path = os.path.join(local_dataset_path, self._COVARIATES_FOLDER_NAME)
        self.metadata_path = os.path.join(local_dataset_path, metadata_file_name)
        metadata = self._load_metadata()
        super().__init__({}, {}, metadata)

    def _load_metadata(self) -> pd.DataFrame:
        """
        Loads metadata from CSV
        """
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        metadata = pd.read_csv(self.metadata_path)
        metadata.set_index(self._INDEX_COL, drop=False, inplace=True)
        return metadata

    def load_series_list(self, series_list: List[str]) -> NoReturn:
        logger.info("Start loading %s series", len(series_list))
        data_dict = {s: self._load_series(s) for s in series_list}
        covariate_dict = {s: self._load_covariates(s) for s in series_list}
        self.dataset.update_data(data_dict, covariate_dict)
        logger.info("Data loading finished.")

    def _load_series(self, series_name: str) -> pd.DataFrame:
        datafile_path = os.path.join(self.local_data_path, series_name)
        if not os.path.exists(datafile_path):
            raise FileNotFoundError(f"Series file not found: {datafile_path}")
        return read_data(datafile_path)

    def _load_covariates(self, series_name: str) -> Optional[Dict]:
        series_name_no_ext = os.path.splitext(series_name)[0]
        cov_path = os.path.join(self.local_covariates_path, series_name_no_ext)
        return read_covariates(cov_path)


class LocalForecastingDataSource(LocalDataSource):
    def __init__(self):
        super().__init__(FORECASTING_DATASET_PATH, "FORECAST_META.csv")


class LocalStForecastingDataSource(LocalDataSource):
    def __init__(self):
        super().__init__(ST_FORECASTING_DATASET_PATH, "ST_FORECAST_META.csv")


class LocalAnomalyDetectDataSource(LocalDataSource):
    def __init__(self):
        super().__init__(ANOMALY_DETECT_DATASET_PATH, "DETECT_META.csv")
