"""Shared feature engineering for the 5G Zone Prediction System.

Every step that derives model features must go through this module so that
inference always computes exactly the features the models were trained on:

- DataExploration/Data_Cleaning.py          -> ProcessedData/clean_data.csv
- DataExploration/CleanData_For_Training.py -> ProcessedData/clean_data_training.csv
- Main/main.py (CSV Zone Heatmap tab)       -> inference-time features

Canonical definitions (verified against the fitted scaler shipped in
TrainedModel/Clustering/cluster_label_scaler.pkl, whose data_max_ for the
five cluster features is [-37.681306, 144.920503, 1464.75, 164.6, 77.84]):

    average_latency  = mean(sv r1..svr4)
    total_throughput = upload_bitrate_mbits/sec + download_bitrate_rx_mbits/sec
    total_bandwidth  = upload_transfer_size_mbytes * download_transfer_size_rx_mbytes

All names below are the canonical (post-rename) column names used by the
processed datasets and expected from CSVs uploaded to the GUI.
"""

# Derived feature columns
AVERAGE_LATENCY = 'average_latency'
TOTAL_THROUGHPUT = 'total_throughput'
TOTAL_BANDWIDTH = 'total_bandwidth'

DERIVED_COLUMNS = [AVERAGE_LATENCY, TOTAL_THROUGHPUT, TOTAL_BANDWIDTH]

# Raw metric columns the derived features are computed from
SVR_COLUMNS = ['svr1', 'svr2', 'svr3', 'svr4']
UPLOAD_BITRATE = 'upload_bitrate_mbits/sec'
DOWNLOAD_BITRATE = 'download_bitrate_rx_mbits/sec'
UPLOAD_TRANSFER_SIZE = 'upload_transfer_size_mbytes'
DOWNLOAD_TRANSFER_SIZE = 'download_transfer_size_rx_mbytes'

RAW_METRIC_COLUMNS = [
    UPLOAD_BITRATE,
    DOWNLOAD_BITRATE,
    UPLOAD_TRANSFER_SIZE,
    DOWNLOAD_TRANSFER_SIZE,
] + SVR_COLUMNS

# Feature vector consumed by the MinMaxScaler + KMeans models, in training order
CLUSTER_FEATURES = ['latitude', 'longitude', AVERAGE_LATENCY, TOTAL_THROUGHPUT, TOTAL_BANDWIDTH]


def missing_columns(df, columns):
    """Return the subset of `columns` that `df` does not contain."""
    return [col for col in columns if col not in df.columns]


def add_derived_features(df, inplace=False):
    """Add average_latency, total_throughput and total_bandwidth to `df`.

    `df` must contain the raw metric columns (RAW_METRIC_COLUMNS) under the
    canonical names. Raises ValueError if any required column is missing.
    """
    missing = missing_columns(df, RAW_METRIC_COLUMNS)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not inplace:
        df = df.copy()

    df[AVERAGE_LATENCY] = df[SVR_COLUMNS].mean(axis=1)
    df[TOTAL_THROUGHPUT] = df[UPLOAD_BITRATE] + df[DOWNLOAD_BITRATE]
    df[TOTAL_BANDWIDTH] = df[UPLOAD_TRANSFER_SIZE] * df[DOWNLOAD_TRANSFER_SIZE]

    return df
