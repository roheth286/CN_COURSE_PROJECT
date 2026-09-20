import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from features.windowing import STATE_FEATURE_NAMES, STATE_VECTOR_DIM
from features.sequence_dataset import (
    split_session_chronologically,
    build_sliding_sequences_for_partition,
    NetworkSequenceDataset,
    NetworkStateScaler,
)


def run_unit_tests() -> None:
    print("=" * 60)
    print("Running Stage 4 Sequence Dataset Unit Tests...")
    print("=" * 60)

    base_time = pd.Timestamp("2017-07-07 09:00:00")
    num_windows = 50
    timestamps_start = [base_time + pd.Timedelta(seconds=30 * i) for i in range(num_windows)]
    timestamps_end = [t + pd.Timedelta(seconds=30) for t in timestamps_start]

    feature_data = {
        name: np.linspace(i + 1, (i + 1) * 10, num_windows)
        for i, name in enumerate(STATE_FEATURE_NAMES)
    }
    feature_data["window_start"] = timestamps_start
    feature_data["window_end"] = timestamps_end
    feature_data["is_attack"] = [0] * 35 + [1] * 15
    feature_data["attack_category"] = ["BENIGN"] * 35 + ["DDoS"] * 15
    feature_data["class_index"] = [0] * 35 + [3] * 15

    session_df = pd.DataFrame(feature_data)

    train_df, val_df, test_df = split_session_chronologically(
        session_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )
    assert len(train_df) == 35
    assert len(val_df) == 7
    assert len(test_df) == 8

    assert train_df["window_end"].max() <= val_df["window_start"].min()
    assert val_df["window_end"].max() <= test_df["window_start"].min()
    print("[PASS] Test 1: Chronological partitioning strictly preserves time boundaries.")

    scaler = NetworkStateScaler()
    train_features = train_df[STATE_FEATURE_NAMES].values
    scaler.fit(train_features)
    assert scaler.mean_.shape[0] == STATE_VECTOR_DIM
    
    norm_train = scaler.transform(train_features)
    assert np.allclose(norm_train.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(norm_train.std(axis=0), 1.0, atol=1e-5)
    print("[PASS] Test 2: Leak-free StandardScaler correctly fitted on train data only.")

    train_df[STATE_FEATURE_NAMES] = norm_train

    m = 10
    K = 5
    x, y_s, y_b, y_c, ts = build_sliding_sequences_for_partition(
        train_df, history_len_m=m, forecast_horizon_k=K, session_name="test_session"
    )

    expected_sequences = 35 - m - K + 1
    assert len(x) == expected_sequences
    assert x.shape == (expected_sequences, m, STATE_VECTOR_DIM)
    assert y_s.shape == (expected_sequences, STATE_VECTOR_DIM)
    assert y_b.shape == (expected_sequences, K)
    assert y_c.shape == (expected_sequences, K)
    print(f"[PASS] Test 3: Sliding sequence shapes verified (X: {x.shape}, Y_state: {y_s.shape}, Y_risk: {y_b.shape}).")

    assert np.allclose(y_s[0], train_df[STATE_FEATURE_NAMES].iloc[m].values)
    print("[PASS] Test 4: World Model target S_{t+m} correctly matches next state vector.")

    dataset = NetworkSequenceDataset(x, y_s, y_b, y_c, timestamps=ts)
    assert len(dataset) == len(x)

    sample_x, sample_y_s, sample_y_b, sample_y_c = dataset[0]
    assert isinstance(sample_x, torch.Tensor)
    assert sample_x.dtype == torch.float32
    assert sample_y_c.dtype == torch.int64
    assert sample_x.shape == (m, STATE_VECTOR_DIM)

    batch_size = 8
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    first_batch = next(iter(loader))
    bx, by_s, by_b, by_c = first_batch
    assert bx.shape == (batch_size, m, STATE_VECTOR_DIM)
    assert by_s.shape == (batch_size, STATE_VECTOR_DIM)
    assert by_b.shape == (batch_size, K)
    assert by_c.shape == (batch_size, K)
    print(f"[PASS] Test 5: PyTorch Dataset and DataLoader batching verified (Batch X: {bx.shape}).")

    print("\n" + "=" * 60)
    print("All Stage 4 Sequence Dataset Unit Tests Passed Successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_unit_tests()
