import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from forecast.rollout import RecursiveForecaster, evaluate_rollout_metrics
from features.sequence_dataset import STATE_FEATURE_NAMES
from data.preprocess import INDEX_TO_CLASS


def run_stage6(
    checkpoint_path: str = "model/checkpoints/best_world_model.pt",
    scaler_path: str = "Datasets/sequences/scaler.pkl",
    test_sequences_path: str = "Datasets/sequences/test_sequences.npz",
    output_dir: str = "Datasets/forecasts",
    K: int = 5,
    batch_size: int = 64,
    device: str = "cpu"
) -> None:
    print("=" * 80)
    print("STAGE 6: MULTI-STEP RECURSIVE FORECASTING EXECUTION")
    print("=" * 80)

    full_checkpoint_path = os.path.join(PROJECT_ROOT, checkpoint_path)
    full_scaler_path = os.path.join(PROJECT_ROOT, scaler_path)
    full_test_path = os.path.join(PROJECT_ROOT, test_sequences_path)
    full_output_dir = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(full_output_dir, exist_ok=True)

    for path_name, p in [
        ("Model Checkpoint", full_checkpoint_path),
        ("Scaler", full_scaler_path),
        ("Test Sequences", full_test_path)
    ]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{path_name} not found at: {p}.")

    print(f"\n[1/5] Loading test sequences from {test_sequences_path}...")
    test_data = np.load(full_test_path)
    x_test = test_data["x"]
    y_state_test = test_data["y_state"]
    y_risk_test = test_data["y_binary"]
    y_class_test = test_data["y_class"]

    num_samples, history_len, state_dim = x_test.shape
    print(f"      Total test sequences   : {num_samples:,}")
    print(f"      Input history shape     : {x_test.shape} ({history_len * 30}s history)")
    print(f"      Target horizon (K)      : {K} steps ({K * 30}s = {K * 0.5:.1f} minutes lookahead)")

    print(f"\n[2/5] Initializing RecursiveForecaster with trained checkpoint...")
    forecaster = RecursiveForecaster(
        model=full_checkpoint_path,
        scaler=full_scaler_path,
        device=device,
        window_seconds=30
    )
    print(f"      Model loaded on device  : {device}")
    print(f"      Scaler loaded           : {forecaster.scaler is not None}")

    print(f"\n[3/5] Executing multi-step closed-loop autoregressive rollouts (K={K})...")
    forecast_result = forecaster.rollout(
        sequences=x_test,
        K=K,
        unscale=True,
        batch_size=batch_size
    )
    print(f"      Rollout complete!")
    print(f"      Predicted states shape  : {forecast_result.predicted_states.shape}")
    print(f"      Risk probabilities shape: {forecast_result.risk_probabilities.shape}")
    print(f"      Category logits shape   : {forecast_result.category_probabilities.shape}")

    print(f"\n[4/5] Evaluating multi-horizon detection performance across horizons...")
    metrics = evaluate_rollout_metrics(
        forecast_result=forecast_result,
        y_risk_true=y_risk_test,
        y_class_true=y_class_test,
        y_state_true=y_state_test,
        risk_threshold=0.5
    )

    print("\n" + "=" * 90)
    print(f"{'HORIZON':<12} | {'LOOKAHEAD':<10} | {'RISK ACC':<10} | {'PRECISION':<10} | {'RECALL':<10} | {'RISK F1':<10} | {'CAT ACC':<10} | {'STATE MSE':<10}")
    print("-" * 90)

    for k_idx in range(K):
        step_num = k_idx + 1
        seconds = forecast_result.horizon_seconds[k_idx]
        key = f"step_{step_num}_{seconds}s"
        m = metrics["horizons"][key]
        mse_str = f"{m['state_mse']:.4f}" if m['state_mse'] is not None else "N/A"
        print(f"Step {step_num:<7} | +{seconds:<4} sec   | {m['risk_accuracy']*100:>6.2f}%    | {m['risk_precision']*100:>6.2f}%    | {m['risk_recall']*100:>6.2f}%    | {m['risk_f1']*100:>6.2f}%    | {m['category_accuracy']*100:>6.2f}%   | {mse_str:>8}")

    print("=" * 90)

    print(f"\n[5/5] Sample Early Warning Attack Forecast Inspection:")
    attack_indices = np.where(y_risk_test[:, 0] == 1)[0]
    if len(attack_indices) > 0:
        sample_idx = int(attack_indices[0])
        sample = forecast_result.get_sequence(sample_idx)
        print(f"      Inspecting Test Sequence #{sample_idx}:")
        print(f"      Ground Truth Attack   : {INDEX_TO_CLASS.get(int(y_class_test[sample_idx, 0]), 'Unknown')}")
        print(f"      Lookahead Horizons    : {sample['horizon_seconds']} seconds")
        print(f"      Predicted Threat Risks: {[round(float(r), 4) for r in sample['risk_probabilities']]}")
        print(f"      Predicted Categories  : {sample['category_names']}")
        
        if sample['predicted_states_unscaled'] is not None:
            unscaled_k1 = sample['predicted_states_unscaled'][0]
            flow_cnt_idx = STATE_FEATURE_NAMES.index("flow_count") if "flow_count" in STATE_FEATURE_NAMES else 0
            pkt_cnt_idx = STATE_FEATURE_NAMES.index("total_packets") if "total_packets" in STATE_FEATURE_NAMES else 3
            print(f"      Physical Telemetry Forecast (k=1):")
            print(f"        - Predicted Active Flows/30s : {unscaled_k1[flow_cnt_idx]:.1f}")
            print(f"        - Predicted Total Packets/30s: {unscaled_k1[pkt_cnt_idx]:.1f}")
    else:
        print("      No attack sequences found in test partition.")

    results_npz_path = os.path.join(full_output_dir, "test_rollout_predictions.npz")
    np.savez_compressed(
        results_npz_path,
        predicted_states=forecast_result.predicted_states,
        predicted_states_unscaled=forecast_result.predicted_states_unscaled,
        risk_probabilities=forecast_result.risk_probabilities,
        category_probabilities=forecast_result.category_probabilities,
        predicted_categories=forecast_result.predicted_categories,
        y_risk_true=y_risk_test,
        y_class_true=y_class_test,
        y_state_true=y_state_test,
        horizon_seconds=np.array(forecast_result.horizon_seconds)
    )
    print(f"\n[Saved] Test rollout predictions saved to: {os.path.join(output_dir, 'test_rollout_predictions.npz')}")

    metrics_json_path = os.path.join(full_output_dir, "horizon_decay_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"[Saved] Horizon decay metrics exported to: {os.path.join(output_dir, 'horizon_decay_metrics.json')}")

    print("\nStage 6 Multi-Step Recursive Forecasting Completed Successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 6: Multi-Step Recursive Rollout Execution")
    parser.add_argument("--checkpoint", type=str, default="model/checkpoints/best_world_model.pt")
    parser.add_argument("--scaler", type=str, default="Datasets/sequences/scaler.pkl")
    parser.add_argument("--test-sequences", type=str, default="Datasets/sequences/test_sequences.npz")
    parser.add_argument("--output-dir", type=str, default="Datasets/forecasts")
    parser.add_argument("--horizon-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    run_stage6(
        checkpoint_path=args.checkpoint,
        scaler_path=args.scaler,
        test_sequences_path=args.test_sequences,
        output_dir=args.output_dir,
        K=args.horizon_k,
        batch_size=args.batch_size,
        device=args.device
    )
