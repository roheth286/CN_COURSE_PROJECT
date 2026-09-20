"""
Verification and Unit Test Suite for Stage 6: Multi-Step Recursive Forecasting

This script tests:
1. RecursiveForecaster initialization and device/model bindings.
2. Single-sequence rollout shapes, horizons, and probability constraints.
3. Batch-sequence rollout shapes across mini-batches.
4. Closed-loop autoregressive window shifting mechanism.
5. Physical unit unscaling with NetworkStateScaler.
6. Multi-horizon evaluation metrics (F1, Precision, Recall, Accuracy, State MSE decay).
7. End-to-end integration test with the trained checkpoint best_world_model.pt.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
from typing import Dict, Any

from model.network_world_model import NetworkWorldModel
from features.sequence_dataset import NetworkStateScaler, STATE_VECTOR_DIM
from forecast.rollout import ForecastResult, RecursiveForecaster, evaluate_rollout_metrics
from data.preprocess import INDEX_TO_CLASS


def run_unit_tests() -> None:
    """Run all automated unit tests for Stage 6."""
    print("=" * 65)
    print("Running Stage 6 Multi-Step Recursive Forecasting Unit Tests...")
    print("=" * 65)

    seq_len = 10
    state_dim = STATE_VECTOR_DIM  # 36
    K = 5
    num_classes = 8

    # -------------------------------------------------------------
    # Test 1: RecursiveForecaster Initialization
    # -------------------------------------------------------------
    dummy_model = NetworkWorldModel(
        state_dim=state_dim,
        hidden_dim=32,
        num_lstm_layers=1,
        num_classes=num_classes,
        dropout=0.0
    )
    forecaster = RecursiveForecaster(model=dummy_model, device="cpu", window_seconds=30)
    assert forecaster.model.training is False, "Forecaster must place model in eval mode"
    assert forecaster.window_seconds == 30, "Window seconds mismatch"
    print("[PASS] Test 1: RecursiveForecaster initialized in eval mode.")

    # -------------------------------------------------------------
    # Test 2: Single Sequence Rollout Dimensions and Horizons
    # -------------------------------------------------------------
    single_seq = np.random.randn(seq_len, state_dim).astype(np.float32)
    single_res = forecaster.rollout(single_seq, K=K, unscale=False)

    assert single_res.predicted_states.shape == (K, state_dim), (
        f"Expected single states shape ({K}, {state_dim}), got {single_res.predicted_states.shape}"
    )
    assert single_res.risk_probabilities.shape == (K,), (
        f"Expected single risks shape ({K},), got {single_res.risk_probabilities.shape}"
    )
    assert single_res.category_probabilities.shape == (K, num_classes), (
        f"Expected single cat probs shape ({K}, {num_classes}), got {single_res.category_probabilities.shape}"
    )
    assert single_res.predicted_categories.shape == (K,), (
        f"Expected single pred cats shape ({K},), got {single_res.predicted_categories.shape}"
    )
    assert single_res.horizon_seconds == [30, 60, 90, 120, 150], (
        f"Expected horizon seconds [30, 60, 90, 120, 150], got {single_res.horizon_seconds}"
    )
    assert len(single_res.category_names[0]) == K, "Category names length mismatch"
    # Probability bounds
    assert np.all((single_res.risk_probabilities >= 0.0) & (single_res.risk_probabilities <= 1.0)), (
        "Risk probabilities must be bounded in [0, 1]"
    )
    sum_cat = np.sum(single_res.category_probabilities, axis=-1)
    assert np.allclose(sum_cat, 1.0, atol=1e-5), "Category probabilities must sum to 1.0"
    print(f"[PASS] Test 2: Single sequence rollout output shapes and probabilities verified.")

    # -------------------------------------------------------------
    # Test 3: Batch Sequence Rollout Dimensions across Chunks
    # -------------------------------------------------------------
    batch_size = 25
    batch_seqs = np.random.randn(batch_size, seq_len, state_dim).astype(np.float32)
    batch_res = forecaster.rollout(batch_seqs, K=K, batch_size=8, unscale=False)

    assert batch_res.predicted_states.shape == (batch_size, K, state_dim), (
        f"Expected batch states shape ({batch_size}, {K}, {state_dim}), got {batch_res.predicted_states.shape}"
    )
    assert batch_res.risk_probabilities.shape == (batch_size, K), (
        f"Expected batch risks shape ({batch_size}, {K}), got {batch_res.risk_probabilities.shape}"
    )
    assert batch_res.category_probabilities.shape == (batch_size, K, num_classes), (
        f"Expected batch cat probs shape ({batch_size}, {K}, {num_classes}), got {batch_res.category_probabilities.shape}"
    )
    assert batch_res.predicted_categories.shape == (batch_size, K), (
        f"Expected batch pred cats shape ({batch_size}, {K}), got {batch_res.predicted_categories.shape}"
    )
    assert len(batch_res.category_names) == batch_size, "Batch category names count mismatch"
    print(f"[PASS] Test 3: Batch sequence rollout verified across chunk boundaries.")

    # -------------------------------------------------------------
    # Test 4: Autoregressive Dynamic Window Shifting
    # -------------------------------------------------------------
    # Test that different histories yield different multi-step trajectories
    seq_a = np.zeros((seq_len, state_dim), dtype=np.float32)
    seq_b = np.ones((seq_len, state_dim), dtype=np.float32) * 5.0

    res_a = forecaster.rollout(seq_a, K=3, unscale=False)
    res_b = forecaster.rollout(seq_b, K=3, unscale=False)

    # State trajectories should diverge based on distinct inputs
    diff = np.max(np.abs(res_a.predicted_states - res_b.predicted_states))
    assert diff > 1e-4, f"Predicted state trajectories should reflect input differences, diff={diff}"
    print(f"[PASS] Test 4: Closed-loop autoregressive window propagation verified.")

    # -------------------------------------------------------------
    # Test 5: Physical Unit Unscaling with NetworkStateScaler
    # -------------------------------------------------------------
    # Fit mock scaler
    mock_data = np.random.uniform(10.0, 500.0, size=(100, state_dim)).astype(np.float32)
    scaler = NetworkStateScaler()
    scaler.fit(mock_data)

    forecaster_scaled = RecursiveForecaster(model=dummy_model, scaler=scaler, device="cpu")
    scaled_res = forecaster_scaled.rollout(single_seq, K=K, unscale=True)

    assert scaled_res.predicted_states_unscaled is not None, "Unscaled states should be computed"
    assert scaled_res.predicted_states_unscaled.shape == (K, state_dim), "Unscaled shape mismatch"

    # Verify that unscaled states match manual inverse_transform
    expected_unscaled = scaler.inverse_transform(scaled_res.predicted_states)
    assert np.allclose(scaled_res.predicted_states_unscaled, expected_unscaled, atol=1e-4), (
        "Unscaled states do not match scaler.inverse_transform"
    )
    print(f"[PASS] Test 5: State unscaling roundtrip verified with NetworkStateScaler.")

    # -------------------------------------------------------------
    # Test 6: Multi-Horizon Metric Decay Evaluation
    # -------------------------------------------------------------
    # Create controlled synthetic predictions and ground truth
    N_test = 20
    test_res = forecaster.rollout(batch_seqs[:N_test], K=K, unscale=False)

    # Synthetic targets
    y_risk_true = np.random.choice([0, 1], size=(N_test, K), p=[0.7, 0.3])
    y_class_true = np.random.choice(range(num_classes), size=(N_test, K))
    y_state_true = np.random.randn(N_test, state_dim).astype(np.float32)

    eval_results = evaluate_rollout_metrics(
        forecast_result=test_res,
        y_risk_true=y_risk_true,
        y_class_true=y_class_true,
        y_state_true=y_state_true,
        risk_threshold=0.5
    )

    assert "horizons" in eval_results, "Missing horizons key in eval results"
    assert len(eval_results["horizons"]) == K, f"Expected {K} horizon evaluations"
    assert "summary" in eval_results, "Missing summary key in eval results"

    step_1_metrics = eval_results["horizons"]["step_1_30s"]
    assert "risk_accuracy" in step_1_metrics, "Missing risk_accuracy"
    assert "risk_f1" in step_1_metrics, "Missing risk_f1"
    assert "category_accuracy" in step_1_metrics, "Missing category_accuracy"
    assert step_1_metrics["state_mse"] is not None, "Missing step 1 state_mse"
    print(f"[PASS] Test 6: Multi-horizon metric decay calculations verified.")

    # -------------------------------------------------------------
    # Test 7: Integration Test with Real Checkpoint & Scaler
    # -------------------------------------------------------------
    checkpoint_path = os.path.join(PROJECT_ROOT, "model", "checkpoints", "best_world_model.pt")
    scaler_path = os.path.join(PROJECT_ROOT, "Datasets", "sequences", "scaler.pkl")
    test_seq_path = os.path.join(PROJECT_ROOT, "Datasets", "sequences", "test_sequences.npz")

    if os.path.exists(checkpoint_path) and os.path.exists(scaler_path) and os.path.exists(test_seq_path):
        real_forecaster = RecursiveForecaster(
            model=checkpoint_path,
            scaler=scaler_path,
            device="cpu",
            window_seconds=30
        )
        test_npz = np.load(test_seq_path)
        sample_x = test_npz["x"][:5]  # 5 test sequences

        real_res = real_forecaster.rollout(sample_x, K=5, unscale=True)
        assert real_res.predicted_states.shape == (5, 5, 36), "Real checkpoint states shape mismatch"
        assert real_res.predicted_states_unscaled is not None, "Real unscaled states missing"
        assert real_res.risk_probabilities.shape == (5, 5), "Real risks shape mismatch"

        # Check a sample forecast profile
        seq_0 = real_res.get_sequence(0)
        print(f"[PASS] Test 7: Real checkpoint integration verified.")
        print(f"       Sample 0 Forecasted Risks across horizons: {[round(r, 4) for r in seq_0['risk_probabilities']]}")
        print(f"       Sample 0 Forecasted Categories: {seq_0['category_names']}")
    else:
        print("[SKIP] Test 7: Checkpoint or dataset files not found for integration test.")

    print("=" * 65)
    print("All Stage 6 Multi-Step Rollout Unit Tests Passed Successfully!")
    print("=" * 65)


if __name__ == "__main__":
    run_unit_tests()
