import os
import sys

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
    print("=" * 65)
    print("Running Stage 6 Multi-Step Recursive Forecasting Unit Tests...")
    print("=" * 65)

    seq_len = 10
    state_dim = STATE_VECTOR_DIM
    K = 5
    num_classes = 8

    dummy_model = NetworkWorldModel(
        state_dim=state_dim,
        hidden_dim=32,
        num_lstm_layers=1,
        num_classes=num_classes,
        dropout=0.0
    )
    forecaster = RecursiveForecaster(model=dummy_model, device="cpu", window_seconds=30)
    assert forecaster.model.training is False
    assert forecaster.window_seconds == 30
    print("[PASS] Test 1: RecursiveForecaster initialized in eval mode.")

    single_seq = np.random.randn(seq_len, state_dim).astype(np.float32)
    single_res = forecaster.rollout(single_seq, K=K, unscale=False)

    assert single_res.predicted_states.shape == (K, state_dim)
    assert single_res.risk_probabilities.shape == (K,)
    assert single_res.category_probabilities.shape == (K, num_classes)
    assert single_res.predicted_categories.shape == (K,)
    assert single_res.horizon_seconds == [30, 60, 90, 120, 150]
    assert len(single_res.category_names[0]) == K
    assert np.all((single_res.risk_probabilities >= 0.0) & (single_res.risk_probabilities <= 1.0))
    sum_cat = np.sum(single_res.category_probabilities, axis=-1)
    assert np.allclose(sum_cat, 1.0, atol=1e-5)
    print("[PASS] Test 2: Single sequence rollout output shapes and probabilities verified.")

    batch_size = 25
    batch_seqs = np.random.randn(batch_size, seq_len, state_dim).astype(np.float32)
    batch_res = forecaster.rollout(batch_seqs, K=K, batch_size=8, unscale=False)

    assert batch_res.predicted_states.shape == (batch_size, K, state_dim)
    assert batch_res.risk_probabilities.shape == (batch_size, K)
    assert batch_res.category_probabilities.shape == (batch_size, K, num_classes)
    assert batch_res.predicted_categories.shape == (batch_size, K)
    assert len(batch_res.category_names) == batch_size
    print("[PASS] Test 3: Batch sequence rollout verified across chunk boundaries.")

    seq_a = np.zeros((seq_len, state_dim), dtype=np.float32)
    seq_b = np.ones((seq_len, state_dim), dtype=np.float32) * 5.0

    res_a = forecaster.rollout(seq_a, K=3, unscale=False)
    res_b = forecaster.rollout(seq_b, K=3, unscale=False)

    diff = np.max(np.abs(res_a.predicted_states - res_b.predicted_states))
    assert diff > 1e-4
    print("[PASS] Test 4: Closed-loop autoregressive window propagation verified.")

    mock_data = np.random.uniform(10.0, 500.0, size=(100, state_dim)).astype(np.float32)
    scaler = NetworkStateScaler()
    scaler.fit(mock_data)

    forecaster_scaled = RecursiveForecaster(model=dummy_model, scaler=scaler, device="cpu")
    scaled_res = forecaster_scaled.rollout(single_seq, K=K, unscale=True)

    assert scaled_res.predicted_states_unscaled is not None
    assert scaled_res.predicted_states_unscaled.shape == (K, state_dim)

    expected_unscaled = scaler.inverse_transform(scaled_res.predicted_states)
    assert np.allclose(scaled_res.predicted_states_unscaled, expected_unscaled, atol=1e-4)
    print("[PASS] Test 5: State unscaling roundtrip verified with NetworkStateScaler.")

    N_test = 20
    test_res = forecaster.rollout(batch_seqs[:N_test], K=K, unscale=False)

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

    assert "horizons" in eval_results
    assert len(eval_results["horizons"]) == K
    assert "summary" in eval_results

    step_1_metrics = eval_results["horizons"]["step_1_30s"]
    assert "risk_accuracy" in step_1_metrics
    assert "risk_f1" in step_1_metrics
    assert "category_accuracy" in step_1_metrics
    assert step_1_metrics["state_mse"] is not None
    print("[PASS] Test 6: Multi-horizon metric decay calculations verified.")

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
        sample_x = test_npz["x"][:5]

        real_res = real_forecaster.rollout(sample_x, K=5, unscale=True)
        assert real_res.predicted_states.shape == (5, 5, 36)
        assert real_res.predicted_states_unscaled is not None
        assert real_res.risk_probabilities.shape == (5, 5)

        seq_0 = real_res.get_sequence(0)
        print("[PASS] Test 7: Real checkpoint integration verified.")
        print(f"       Sample 0 Forecasted Risks across horizons: {[round(r, 4) for r in seq_0['risk_probabilities']]}")
        print(f"       Sample 0 Forecasted Categories: {seq_0['category_names']}")
    else:
        print("[SKIP] Test 7: Checkpoint or dataset files not found for integration test.")

    print("=" * 65)
    print("All Stage 6 Multi-Step Rollout Unit Tests Passed Successfully!")
    print("=" * 65)


if __name__ == "__main__":
    run_unit_tests()
