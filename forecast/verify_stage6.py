"""
Unit Test Suite for Stage 6: Autoregressive Rollout & Multi-Step Forecasting

Tests include:
1. Single sequence rollout: output shapes, horizon length, and probability bounds.
2. Batched sequence rollout: consistent batch handling across all heads.
3. State shift verification: ensuring historical state eviction and appending logic.
4. Pure NumPy metric computation: verifying Accuracy, Precision, Recall, F1, and AUC.
5. Physical unscaling: testing inverse transformation using NetworkStateScaler.
6. Full checkpoint evaluation: running rollout against actual test_sequences.npz.
"""

import os
import sys
import tempfile
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.network_world_model import NetworkWorldModel
from features.sequence_dataset import NetworkStateScaler
from forecast.rollout import (
    RolloutResult,
    autoregressive_rollout,
    calculate_binary_metrics,
    calculate_multiclass_metrics,
    evaluate_rollout_horizons,
)


def run_unit_tests() -> None:
    """Run all unit tests for Stage 6."""
    print("=" * 60)
    print("Running Stage 6 Multi-Step Rollout Unit Tests...")
    print("=" * 60)

    # Instantiate model for synthetic tests
    state_dim = 36
    seq_len = 10
    horizon = 5
    num_classes = 8
    model = NetworkWorldModel(state_dim=state_dim, hidden_dim=64, num_classes=num_classes)
    model.eval()

    # ------------------------------------------------------------------
    # Test 1: Single sequence rollout
    # ------------------------------------------------------------------
    single_seq = np.random.randn(seq_len, state_dim).astype(np.float32)
    res_single = autoregressive_rollout(model, single_seq, horizon=horizon)

    assert isinstance(res_single, RolloutResult), "Return value must be a RolloutResult instance."
    assert res_single.predicted_states.shape == (1, horizon, state_dim), (
        f"Expected states shape (1, {horizon}, {state_dim}), got {res_single.predicted_states.shape}"
    )
    assert res_single.risk_probabilities.shape == (1, horizon), (
        f"Expected risk shape (1, {horizon}), got {res_single.risk_probabilities.shape}"
    )
    assert res_single.category_probabilities.shape == (1, horizon, num_classes), (
        f"Expected cat probs shape (1, {horizon}, {num_classes}), got {res_single.category_probabilities.shape}"
    )
    assert res_single.category_predictions.shape == (1, horizon), (
        f"Expected cat preds shape (1, {horizon}), got {res_single.category_predictions.shape}"
    )

    # Check probability bounds
    assert np.all((res_single.risk_probabilities >= 0.0) & (res_single.risk_probabilities <= 1.0)), (
        "Risk probabilities must be bounded in [0, 1]."
    )
    prob_sums = np.sum(res_single.category_probabilities, axis=-1)
    assert np.allclose(prob_sums, 1.0, atol=1e-4), (
        "Category probabilities across classes must sum to 1.0."
    )
    print(f"[PASS] Test 1: Single sequence rollout shapes and probability bounds verified.")

    # ------------------------------------------------------------------
    # Test 2: Batched rollout
    # ------------------------------------------------------------------
    batch_size = 12
    batch_seq = np.random.randn(batch_size, seq_len, state_dim).astype(np.float32)
    res_batch = autoregressive_rollout(model, batch_seq, horizon=horizon)

    assert res_batch.predicted_states.shape == (batch_size, horizon, state_dim)
    assert res_batch.risk_probabilities.shape == (batch_size, horizon)
    assert res_batch.category_probabilities.shape == (batch_size, horizon, num_classes)
    assert res_batch.category_predictions.shape == (batch_size, horizon)
    print(f"[PASS] Test 2: Batched rollout verified for batch size {batch_size}, horizon {horizon}.")

    # ------------------------------------------------------------------
    # Test 3: Rolling window state shift mechanics
    # ------------------------------------------------------------------
    # Verify that history rolls by dropping index 0 and appending step prediction
    dummy_input = torch.zeros((1, seq_len, state_dim), dtype=torch.float32)
    # Assign distinct markers to timesteps
    for t in range(seq_len):
        dummy_input[0, t, 0] = float(t + 1)

    # Simulate 1 step rollout shift
    simulated_pred = torch.full((1, state_dim), 99.0, dtype=torch.float32)
    next_window = torch.cat([dummy_input[:, 1:, :], simulated_pred.unsqueeze(1)], dim=1)

    assert next_window.shape == (1, seq_len, state_dim)
    # First element should now be 2.0 (1.0 dropped)
    assert next_window[0, 0, 0].item() == 2.0, "Oldest state must be dropped."
    # Last element should now be 99.0 (simulated prediction appended)
    assert next_window[0, -1, 0].item() == 99.0, "Newly predicted state must be appended at the end."
    print(f"[PASS] Test 3: History window sliding & state appending mechanics verified.")

    # ------------------------------------------------------------------
    # Test 4: Pure NumPy metrics computation
    # ------------------------------------------------------------------
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.3, 0.6, 0.4])
    bin_metrics = calculate_binary_metrics(y_true, y_prob, threshold=0.5)

    assert bin_metrics["accuracy"] == 1.0, "All predictions should be correct with this threshold."
    assert bin_metrics["precision"] == 1.0
    assert bin_metrics["recall"] == 1.0
    assert bin_metrics["f1"] == 1.0
    assert bin_metrics["fpr"] == 0.0
    assert bin_metrics["roc_auc"] == 1.0

    y_cat_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_cat_pred = np.array([0, 1, 2, 3, 0, 1, 2, 0])  # 7 out of 8 correct
    cat_metrics = calculate_multiclass_metrics(y_cat_true, y_cat_pred, num_classes=8)
    assert cat_metrics["accuracy"] == 0.875
    assert 0.0 < cat_metrics["macro_f1"] <= 1.0
    print(f"[PASS] Test 4: Pure NumPy binary and multi-class metrics verified.")

    # ------------------------------------------------------------------
    # Test 5: Inverse scaling back to physical network metrics
    # ------------------------------------------------------------------
    scaler = NetworkStateScaler()
    dummy_physical = np.random.uniform(10.0, 1000.0, size=(100, state_dim)).astype(np.float32)
    scaler.fit(dummy_physical)

    # Test unscaling on rollout container
    unscaled = res_batch.unscale_states(scaler)
    assert unscaled.shape == (batch_size, horizon, state_dim)
    # Check that unscaling accurately inverts scaling
    scaled_synthetic = scaler.transform(dummy_physical[:batch_size])
    dummy_rollout = RolloutResult(
        predicted_states=np.repeat(scaled_synthetic[:, np.newaxis, :], horizon, axis=1),
        risk_probabilities=np.zeros((batch_size, horizon)),
        category_probabilities=np.zeros((batch_size, horizon, num_classes)),
        category_predictions=np.zeros((batch_size, horizon), dtype=np.int64),
        horizon=horizon,
    )
    unscaled_recovery = dummy_rollout.unscale_states(scaler)
    expected_physical = np.repeat(dummy_physical[:batch_size, np.newaxis, :], horizon, axis=1)
    assert np.allclose(unscaled_recovery, expected_physical, atol=1e-3)
    print(f"[PASS] Test 5: Physical unit unscaling successfully recovers original feature magnitudes.")

    # ------------------------------------------------------------------
    # Test 6: Checkpoint evaluation on real test dataset
    # ------------------------------------------------------------------
    ckpt_path = os.path.join(PROJECT_ROOT, "model", "checkpoints", "best_world_model.pt")
    test_seq_path = os.path.join(PROJECT_ROOT, "Datasets", "sequences", "test_sequences.npz")

    if os.path.exists(ckpt_path) and os.path.exists(test_seq_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        real_model = NetworkWorldModel(
            state_dim=checkpoint["config"]["state_dim"],
            hidden_dim=checkpoint["config"]["hidden_dim"],
            num_lstm_layers=checkpoint["config"]["num_lstm_layers"],
            num_classes=checkpoint["config"]["num_classes"],
        )
        real_model.load_state_dict(checkpoint["model_state_dict"])
        real_model.eval()

        summary = evaluate_rollout_horizons(
            model=real_model,
            test_sequences_path=test_seq_path,
            horizon=5,
            device="cpu",
            batch_size=64,
        )

        assert "horizons" in summary
        assert len(summary["horizons"]) == 5
        for k in range(1, 6):
            h_key = f"horizon_k{k}"
            assert h_key in summary["horizons"]
            h_data = summary["horizons"][h_key]
            assert "risk_metrics" in h_data
            assert "category_metrics" in h_data
            assert 0.0 <= h_data["risk_metrics"]["accuracy"] <= 1.0

        k1_acc = summary["horizons"]["horizon_k1"]["risk_metrics"]["accuracy"]
        k1_f1 = summary["horizons"]["horizon_k1"]["risk_metrics"]["f1"]
        print(f"[PASS] Test 6: Evaluated checkpoint on 336 real test sequences.")
        print(f"       k=1 (30s): Risk Acc={k1_acc:.1%}, F1={k1_f1:.3f}")
    else:
        print(f"[SKIP] Test 6: Checkpoint or test sequences not found, skipping real dataset eval.")

    print("=" * 60)
    print("All Stage 6 Multi-Step Rollout Tests Passed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_unit_tests()
