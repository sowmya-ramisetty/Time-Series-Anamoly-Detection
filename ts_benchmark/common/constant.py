# -*- coding: utf-8 -*-
import os

# Get the root path where the code file is located
ROOT_PATH = os.path.abspath(os.path.join(__file__, "..", "..", ".."))

# ✅ Fix: added 'datasets' before 'dataset'
FORECASTING_DATASET_PATH = os.path.join(ROOT_PATH, "datasets", "dataset", "forecasting")
ST_FORECASTING_DATASET_PATH = os.path.join(ROOT_PATH, "datasets", "dataset", "st_forecasting")
ANOMALY_DETECT_DATASET_PATH = os.path.join(ROOT_PATH, "datasets", "dataset", "anomaly_detect")

# Profile Path
CONFIG_PATH = os.path.join(ROOT_PATH, "config")

# third-party library path
THIRD_PARTY_PATH = os.path.join(ROOT_PATH, "ts_benchmark", "baselines", "third_party")
