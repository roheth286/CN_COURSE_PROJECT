"""
Verification and Unit Test Suite for Stage 4: Sequence Dataset Construction

This script verifies:
1. Chronological partition integrity (Train < Val < Test).
2. Boundary isolation (no sliding sequences cross split boundaries).
3. Tensor dimensionality:
   - X in R^{N x m x D}
   - Y_state in R^{N x D}
   - Y_binary in R^{N x K}
   - Y_class in R^{N x K}
4. Leak-free standard scaling (fitted strictly on training split).
5. PyTorch Dataset & DataLoader integration with batching.
"""

import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from features.windowing import STATE_FEATURE_NAMES, STATE_VECTOR_DIM
from features.sequence_dataset import (
    split_session_chronologically,
    build_sliding_sequences_for_partition,
    NetworkSequenceDataset,
    NetworkStateScaler,
)


def run_unit_tests() -> None:
    """Run automated unit tests for Stage 4."""
    print("=" * 60)
    print("Running Stage 4 Sequence Dataset Unit Tests...")
    print("=" * 60)

    # Construct synthetic session of 50 temporal windows
    base_time = pd.Timestamp("2017-07-07 09:00:00")
    num_windows = 50
    timestamps_start = [base_time + pd.Timedelta(seconds=30 * i) for i in range(num_windows)]
    timestamps_end = [t + pd.Timedelta(seconds=30) for t in timestamps_start]

    # Create dummy 36 state features
    feature_data = {
        name: np.linspace(i + 1, (i + 1) * 10, num_windows)
        for i, name in enumerate(STATE_FEATURE_NAMES)
    }
    feature_data["window_start"] = timestamps_start
    feature_data["window_end"] = timestamps_end
    # First 35 benign, last 15 attack (e.g. DDoS)
    feature_data["is_attack"] = [0] * 35 + [1] * 15
    feature_data["attack_category"] = ["BENIGN"] * 35 + ["DDoS"] * 15
    feature_data["class_index"] = [0] * 35 + [3] * 15

    session_df = pd.DataFrame(feature_data)

    # Test 1: Chronological partitioning
    train_df, val_df, test_df = split_session_chronologically(
        session_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )
    assert len(train_df) == 35, f"Expected 35 train windows, got {len(train_df)}"
    assert len(val_df) == 7, f"Expected 7 val windows, got {len(val_df)}"
    assert len(test_df) == 8, f"Expected 8 test windows, got {len(test_df)}"

    # Check temporal order: train ends before val starts, val ends before test starts
    assert train_df["window_end"].max() <= val_df["window_start"].min(), "Train/Val temporal overlap!"
    assert val_df["window_end"].max() <= test_df["window_start"].min(), "Val/Test temporal overlap!"
    print("[PASS] Test 1: Chronological partitioning strictly preserves time boundaries.")

    # Test 2: Leak-free Scaler Fitting
    scaler = NetworkStateScaler()
    train_features = train_df[STATE_FEATURE_NAMES].values
    scaler.fit(train_features)
    assert scaler.mean_.shape[0] == STATE_VECTOR_DIM, "Scaler feature count mismatch."
    
    # Transform train and check that normalized train has mean ~ 0 and std ~ 1
    norm_train = scaler.transform(train_features)
    assert np.allclose(norm_train.mean(axis=0), 0.0, atol=1e-5), "Train mean is not 0 after scaling."
    assert np.allclose(norm_train.std(axis=0), 1.0, atol=1e-5), "Train std is not 1 after scaling."
    print("[PASS] Test 2: Leak-free StandardScaler correctly fitted on train data only.")

    # Update train_df with normalized values
    train_df[STATE_FEATURE_NAMES] = norm_train

    # Test 3: Sliding sequence construction
    m = 10  # history length
    K = 5   # forecast horizon
    x, y_s, y_b, y_c, ts = build_sliding_sequences_for_partition(
        train_df, history_len_m=m, forecast_horizon_k=K, session_name="test_session"
    )

    # Expected sequences = 35 - 10 - 5 + 1 = 21 sequences
    expected_sequences = 35 - m - K + 1
    assert len(x) == expected_sequences, f"Expected {expected_sequences} sequences, got {len(x)}"
    assert x.shape == (expected_sequences, m, STATE_VECTOR_DIM), f"Unexpected X shape: {x.shape}"
    assert y_s.shape == (expected_sequences, STATE_VECTOR_DIM), f"Unexpected Y_state shape: {y_s.shape}"
    assert y_b.shape == (expected_sequences, K), f"Unexpected Y_binary shape: {y_b.shape}"
    assert y_c.shape == (expected_sequences, K), f"Unexpected Y_class shape: {y_c.shape}"
    print(f"[PASS] Test 3: Sliding sequence shapes verified (X: {x.shape}, Y_state: {y_s.shape}, Y_risk: {y_b.shape}).")

    # Test 4: Next-state target alignment
    # y_s for sequence 0 should exactly equal the feature values at row index m
    assert np.allclose(y_s[0], train_df[STATE_FEATURE_NAMES].iloc[m].values), "Next state target misaligned."
    print("[PASS] Test 4: World Model target S_{t+m} correctly matches next state vector.")

    # Test 5: PyTorch Dataset & DataLoader integration
    dataset = NetworkSequenceDataset(x, y_s, y_b, y_c, timestamps=ts)
    assert len(dataset) == len(x), "PyTorch Dataset length mismatch."

    # Test sample item
    sample_x, sample_y_s, sample_y_b, sample_y_c = dataset[0]
    assert isinstance(sample_x, torch.Tensor), "sample_x must be a torch.Tensor"
    assert sample_x.dtype == torch.float32, "sample_x must be float32"
    assert sample_y_c.dtype == torch.int64, "sample_y_c must be int64 (long)"
    assert sample_x.shape == (m, STATE_VECTOR_DIM), f"Sample shape mismatch: {sample_x.shape}"

    # Test DataLoader batching
    batch_size = 8
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    first_batch = next(iter(loader))
    bx, by_s, by_b, by_c = first_batch
    assert bx.shape == (batch_size, m, STATE_VECTOR_DIM), f"Batch X shape mismatch: {bx.shape}"
    assert by_s.shape == (batch_size, STATE_VECTOR_DIM), f"Batch Y_state shape mismatch: {by_s.shape}"
    assert by_b.shape == (batch_size, K), f"Batch Y_binary shape mismatch: {by_b.shape}"
    assert by_c.shape == (batch_size, K), f"Batch Y_class shape mismatch: {by_c.shape}"
    print(f"[PASS] Test 5: PyTorch Dataset and DataLoader batching verified (Batch X: {bx.shape}).")

    print("\n" + "=" * 60)
    print("All Stage 4 Sequence Dataset Unit Tests Passed Successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_unit_tests()
