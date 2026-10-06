"""Regression tests: inference-time features must equal training-time features.

Locks in the fix for the bug where Main/main.py derived
total_throughput/total_bandwidth with different formulas than the ones the
KMeans/scaler artifacts were trained with (definitions live in features.py).

Run from anywhere:

    py -m pytest tests/test_feature_parity.py -q
    (or) py tests/test_feature_parity.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Main'))
import main as gui  # imports tkinter/matplotlib but creates no window

from features import (CLUSTER_FEATURES, DERIVED_COLUMNS, RAW_METRIC_COLUMNS,
                      add_derived_features)

TRAIN_CSV = ROOT / 'ProcessedData' / 'clean_data_training.csv'
CLUSTER_OUTPUT_CSV = ROOT / 'TrainedModel' / 'Clustering' / 'clustering_output.csv'
TOL = 1e-9


def _require(*paths):
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing data artifacts (run the cleaning/training scripts first): {missing}")


def test_missing_columns_raise():
    df = pd.DataFrame({'latitude': [1.0], 'svr1': [1.0]})
    try:
        add_derived_features(df)
    except ValueError as exc:
        assert 'Missing required columns' in str(exc)
    else:
        raise AssertionError('expected ValueError for missing columns')


def test_cluster_features_match_saved_artifacts():
    _require(ROOT / 'TrainedModel' / 'Clustering' / 'cluster_label_scaler.pkl',
             ROOT / 'TrainedModel' / 'Clustering' / 'cluster_label_kmeans.pkl')
    import joblib
    scaler_bundle = joblib.load(ROOT / 'TrainedModel' / 'Clustering' / 'cluster_label_scaler.pkl')
    kmeans_bundle = joblib.load(ROOT / 'TrainedModel' / 'Clustering' / 'cluster_label_kmeans.pkl')
    assert scaler_bundle['features'] == CLUSTER_FEATURES, \
        f"scaler trained on {scaler_bundle['features']}, module declares {CLUSTER_FEATURES}"
    assert kmeans_bundle['features'] == CLUSTER_FEATURES


def test_add_derived_features_matches_training_data():
    """The shared module must reproduce the derived columns of the training CSV."""
    _require(TRAIN_CSV)
    cols = ['latitude', 'longitude'] + RAW_METRIC_COLUMNS + DERIVED_COLUMNS
    df = pd.read_csv(TRAIN_CSV, usecols=cols)

    stored = df[DERIVED_COLUMNS]
    recomputed = add_derived_features(df.drop(columns=DERIVED_COLUMNS))
    diff = (recomputed[DERIVED_COLUMNS] - stored).abs().max()
    assert (diff < TOL).all(), f"derived feature mismatch vs training CSV: {diff.to_dict()}"


def test_gui_assign_zones_reproduces_training_clusters():
    """Rows the GUI sees must get the same zone as the training run produced."""
    _require(TRAIN_CSV, CLUSTER_OUTPUT_CSV)
    sys.path.insert(0, str(ROOT / 'Main'))

    system = gui.ZoneForecastSystem(
        str(ROOT / 'TrainedModel' / 'Clustering' / 'cluster_label_kmeans.pkl'),
        str(ROOT / 'TrainedModel' / 'Clustering' / 'cluster_label_scaler.pkl'),
        str(ROOT / 'TrainedModel' / 'Clustering' / 'clustering_output.csv'),
        str(ROOT / 'TrainedModel' / 'TimeSeries' / 'arima_model.pkl'),
        str(ROOT / 'TrainedModel' / 'Clustering' / 'zone_cluster_map.csv'),
    )

    # Feed the GUI only the raw columns: it must derive the rest itself.
    cols = ['latitude', 'longitude'] + RAW_METRIC_COLUMNS
    inferred = system.assign_zones(pd.read_csv(TRAIN_CSV, usecols=cols))
    stored = pd.read_csv(CLUSTER_OUTPUT_CSV, usecols=['cluster', 'performance_label'])

    cluster_mismatch = int((inferred['cluster'].to_numpy() != stored['cluster'].to_numpy()).sum())
    label_mismatch = int((inferred['performance_label'].to_numpy() !=
                          stored['performance_label'].to_numpy()).sum())
    n = len(stored)
    assert cluster_mismatch == 0, f"{cluster_mismatch}/{n} rows get a different cluster"
    assert label_mismatch == 0, f"{label_mismatch}/{n} rows get a different performance label"


if __name__ == '__main__':
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith('test_') and callable(fn)]
    for name, fn in tests:
        fn()
        print(f'PASS {name}')
    print(f'{len(tests)} passed')
