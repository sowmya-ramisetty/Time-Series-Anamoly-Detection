import logging
import os
import pickle
from typing import Union, Any, Optional, Dict

import numpy as np
import pandas as pd
import scipy.sparse

logger = logging.getLogger(__name__)

# Frequency mapping for time series data
FREQ_MAP = {
    "Y": "yearly", "A": "yearly", "M": "monthly", "W": "weekly", "D": "daily", "H": "hourly",
    "Q": "quarterly", "B": "daily", "C": "daily", "UNKNOWN": "other",
}

COVARIATES_LOAD_METHOD = {
    "adj.npz": scipy.sparse.load_npz,
}

# --- Helper Functions ---

def is_st(data: pd.DataFrame) -> bool:
    """Check if data is in spatial-temporal format."""
    return data.shape[1] == 4


def read_covariates(folder_path: str) -> Optional[Dict]:
    """Read covariates from a directory."""
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        return None

    covariates = {}
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            try:
                covariates[filename] = get_covariate(filepath)
            except Exception as e:
                logger.warning("Error reading covariate %s: %s", filename, e)
    return covariates or None


def get_covariate(file_path: str) -> Any:
    """Read a covariate file and return its content."""
    covariate_type = os.path.basename(file_path)
    if covariate_type not in COVARIATES_LOAD_METHOD:
        raise ValueError(f"Unsupported covariate type: {covariate_type}")
    return COVARIATES_LOAD_METHOD[covariate_type](file_path)


# --- Main Data Functions ---

def read_data(path: str, nrows=None) -> Union[pd.DataFrame, np.ndarray]:
    """Read data safely — supports both metadata CSV and time series."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    data = pd.read_csv(path, nrows=nrows)
    logger.info(f"Loaded file: {path} with shape {data.shape}")

    # If it’s metadata like DETECT_META.csv → process differently
    if "file_name" in data.columns and "dataset_name" in data.columns:
        logger.info("Detected metadata file structure. Returning as-is.")
        return process_metadata_df(data, nrows)

    # Otherwise treat as series data
    if is_st(data):
        return process_data_np(data, nrows)
    else:
        return process_data_df(data, nrows)


def process_metadata_df(data: pd.DataFrame, nrows=None) -> pd.DataFrame:
    """Process metadata CSV files like DETECT_META.csv."""
    expected_cols = [
        "file_name", "trend", "seasonal", "stationary", "pattern", "shifting",
        "dataset_name", "train_lens", "test_lens", "time_steps", "if_univariate",
        "size", "type_value", "anomaly_rate"
    ]

    # Fill missing expected columns with None
    for col in expected_cols:
        if col not in data.columns:
            data[col] = None

    # Convert numerics safely
    numeric_cols = ["train_lens", "test_lens", "time_steps", "anomaly_rate"]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Truncate rows if needed
    if nrows is not None and isinstance(nrows, int) and data.shape[0] >= nrows:
        data = data.iloc[:nrows, :]

    logger.info(f"Processed metadata DataFrame with shape {data.shape}")
    return data


def process_data_df(data: pd.DataFrame, nrows=None) -> pd.DataFrame:
    """Handle normal time series CSVs (non-ST)."""
    # ✅ Fix: handle missing 'cols' safely
    if "cols" not in data.columns:
        logger.warning("Column 'cols' not found — skipping label detection.")
        data["cols"] = ["value"] * len(data)

    label_exists = "label" in data["cols"].values
    all_points = data.shape[0]
    columns = data.columns

    if columns[0] == "date":
        n_points = data.iloc[:, 2].value_counts().max()
    else:
        n_points = data.iloc[:, 1].value_counts().max()

    is_univariate = n_points == all_points
    n_cols = all_points // n_points
    df = pd.DataFrame()

    cols_name = data["cols"].unique()

    if columns[0] == "date" and not is_univariate:
        df["date"] = data.iloc[:n_points, 0]
        col_data = {
            cols_name[j]: data.iloc[j * n_points:(j + 1) * n_points, 1].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    elif columns[0] != "date" and not is_univariate:
        col_data = {
            cols_name[j]: data.iloc[j * n_points:(j + 1) * n_points, 0].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)

    elif columns[0] == "date" and is_univariate:
        df["date"] = data.iloc[:, 0]
        df[cols_name[0]] = data.iloc[:, 1]
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    else:
        df[cols_name[0]] = data.iloc[:, 0]

    if label_exists:
        last_col_name = df.columns[-1]
        df.rename(columns={last_col_name: "label"}, inplace=True)

    if nrows is not None and isinstance(nrows, int) and df.shape[0] >= nrows:
        df = df.iloc[:nrows, :]

    return df


def process_data_np(df: pd.DataFrame, nrows=None) -> np.ndarray:
    """Convert spatial-temporal data into numpy array."""
    pivot_df = df.pivot_table(index="date", columns=["id", "cols"], values="data")
    sensors = df["id"].unique()
    features = df["cols"].unique()
    pivot_df = pivot_df.reindex(
        columns=pd.MultiIndex.from_product([sensors, features]), fill_value=np.nan
    )
    data_np = pivot_df.to_numpy().reshape(len(pivot_df), len(sensors), len(features))
    data_np = np.transpose(data_np, (0, 2, 1))
    if nrows is not None:
        data_np = data_np[:nrows, :, :]
    return data_np


def load_series_info(file_path: str) -> dict:
    """Get series info for a dataset."""
    raw_data = pd.read_csv(file_path)
    if not is_st(raw_data):
        data = process_data_df(raw_data)
    else:
        data = process_data_np(raw_data)

    if_univariate = data.shape[1] == 1 if isinstance(data, pd.DataFrame) else False
    length = data.shape[0]
    time_stamp = pd.to_datetime(raw_data.iloc[:length, 0])
    freq = pd.infer_freq(time_stamp)
    freq = FREQ_MAP.get(freq, "other")
    file_name = os.path.basename(file_path)

    return {
        "file_name": file_name,
        "freq": freq,
        "if_univariate": if_univariate,
        "size": "user",
        "length": length,
        "trend": "",
        "seasonal": "",
        "stationary": "",
        "transition": "",
        "shifting": "",
        "correlation": "",
    }
