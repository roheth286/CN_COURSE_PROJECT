import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import argparse
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from forecast.rollout import RecursiveForecaster
from forecast.lead_time import LeadTimeEngine, LeadTimeSummary


def run_stage7(
    checkpoint_path: str = "model/checkpoints/best_world_model.pt",
    scaler_path: str = "Datasets/sequences/scaler.pkl",
    state_vectors_dir: str = "Datasets/state_vectors",
    output_dir: str = "Datasets/forecasts",
    threat_threshold: float = 0.50,
    history_len_m: int = 10,
    K: int = 5,
    device: str = "cpu"
) -> None:
    print("=" * 85)
    print("STAGE 7: ADVANCE WARNING LEAD TIME (DELTA T) EVALUATION")
    print("=" * 85)

    full_checkpoint_path = os.path.join(PROJECT_ROOT, checkpoint_path)
    full_scaler_path = os.path.join(PROJECT_ROOT, scaler_path)
    full_state_dir = os.path.join(PROJECT_ROOT, state_vectors_dir)
    full_output_dir = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(full_output_dir, exist_ok=True)

    print("\n[1/4] Initializing RecursiveForecaster and LeadTimeEngine...")
    forecaster = RecursiveForecaster(
        model=full_checkpoint_path,
        scaler=full_scaler_path,
        device=device,
        window_seconds=30
    )

    engine = LeadTimeEngine(
        forecaster=forecaster,
        threat_threshold=threat_threshold,
        use_composite_threat=True,
        window_seconds=30
    )
    print(f"      Threat Decision Threshold : {threat_threshold:.2f}")
    print(f"      Composite Threat Analysis : Enabled (max[P_risk, 1 - P_benign])")
    print(f"      Sequence History (m)      : {history_len_m} windows ({history_len_m * 30}s)")
    print(f"      Forecast Lookahead (K)    : {K} windows ({K * 30}s = {K * 0.5:.1f} mins)")

    print("\n[2/4] Evaluating Lead Times across full continuous attack episodes...")
    summary_all: LeadTimeSummary = engine.evaluate_all(
        state_vectors_dir=state_vectors_dir,
        history_len_m=history_len_m,
        K=K,
        partition_mode="all"
    )

    print("\n[3/4] Evaluating Lead Times strictly on held-out test partitions...")
    summary_test: LeadTimeSummary = engine.evaluate_all(
        state_vectors_dir=state_vectors_dir,
        history_len_m=history_len_m,
        K=K,
        partition_mode="test"
    )

    print("\n" + "=" * 105)
    print(f"{'EPISODE':<4} | {'SESSION':<24} | {'CATEGORY':<14} | {'START':<19} | {'STATUS':<23} | {'LEAD TIME':<10}")
    print("-" * 105)

    for ep in summary_all.episode_details:
        lt_str = f"+{ep['lead_time_seconds']:.0f}s" if ep['lead_time_seconds'] > 0 else (
            f"{ep['lead_time_seconds']:.0f}s" if ep['detected'] else "N/A"
        )
        start_t = str(ep['start_timestamp'])[:19] if ep['start_timestamp'] else "N/A"
        status_str = ep['detection_type']
        print(f"#{ep['episode_id']:<3} | {ep['session_name'][:24]:<24} | {ep['attack_category'][:14]:<14} | {start_t:<19} | {status_str:<23} | {lt_str:<10}")

    print("=" * 105)

    print("\n" + "=" * 80)
    print("LEAD TIME BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Total Attack Episodes Evaluated   : {summary_all.total_episodes}")
    print(f"Successfully Detected Episodes     : {summary_all.detected_episodes} ({summary_all.detection_rate*100:.1f}%)")
    print(f"  * Proactive Early Warnings (ΔT > 0): {summary_all.early_warning_episodes} ({summary_all.early_warning_rate*100:.1f}% of detected)")
    print(f"  * Onset Detections (ΔT = 0s)      : {summary_all.onset_episodes}")
    print(f"  * Delayed Detections (ΔT < 0s)    : {summary_all.delayed_episodes}")
    print(f"  * Missed Episodes                 : {summary_all.missed_episodes}")
    print(f"Mean Lead Time (Detected Attacks)   : {summary_all.mean_lead_time_seconds:.1f} seconds ({summary_all.mean_lead_time_seconds / 60:.2f} mins)")
    print(f"Median Lead Time                    : {summary_all.median_lead_time_seconds:.1f} seconds")
    print(f"Max Lead Time Achieved              : {summary_all.max_lead_time_seconds:.1f} seconds ({summary_all.max_lead_time_seconds / 60:.2f} mins)")
    print(f"Total Surveillance Windows          : {summary_all.total_benign_windows}")
    print(f"False Alarms                        : {summary_all.false_alarms}")
    print(f"False Alarm Rate                    : {summary_all.false_alarm_rate*100:.2f}%")
    print("-" * 80)

    print("\nPER-CATEGORY ADVANCE WARNING BREAKDOWN:")
    print(f"{'CATEGORY':<16} | {'TOTAL':<6} | {'DETECTED':<9} | {'EARLY WARNED':<13} | {'MEAN LEAD TIME':<15} | {'MAX LEAD TIME':<14}")
    print("-" * 80)
    for cat_name, cat_m in summary_all.category_metrics.items():
        print(f"{cat_name:<16} | {cat_m['total_episodes']:<6} | {cat_m['detected']:<9} | {cat_m['early_warned']:<13} | {cat_m['mean_lead_time_seconds']:>5.1f} sec        | {cat_m['max_lead_time_seconds']:>5.1f} sec")
    print("=" * 80)

    summary_all_path = os.path.join(full_output_dir, "lead_time_all_summary.json")
    with open(summary_all_path, "w", encoding="utf-8") as f:
        json.dump(summary_all.to_dict(), f, indent=4)
    print(f"\n[Saved] All-session lead time summary exported to: {os.path.join(output_dir, 'lead_time_all_summary.json')}")

    summary_test_path = os.path.join(full_output_dir, "lead_time_test_summary.json")
    with open(summary_test_path, "w", encoding="utf-8") as f:
        json.dump(summary_test.to_dict(), f, indent=4)
    print(f"[Saved] Test-partition lead time summary exported to: {os.path.join(output_dir, 'lead_time_test_summary.json')}")

    episodes_df = pd.DataFrame(summary_all.episode_details)
    episodes_csv_path = os.path.join(full_output_dir, "lead_time_episodes.csv")
    episodes_df.to_csv(episodes_csv_path, index=False)
    print(f"[Saved] Episode details CSV exported to: {os.path.join(output_dir, 'lead_time_episodes.csv')}")

    print("\nStage 7 Early Warning Lead Time Engine Completed Successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 7: Early Warning Lead Time Execution")
    parser.add_argument("--checkpoint", type=str, default="model/checkpoints/best_world_model.pt")
    parser.add_argument("--scaler", type=str, default="Datasets/sequences/scaler.pkl")
    parser.add_argument("--state-dir", type=str, default="Datasets/state_vectors")
    parser.add_argument("--output-dir", type=str, default="Datasets/forecasts")
    parser.add_argument("--threat-threshold", type=float, default=0.50)
    parser.add_argument("--history-m", type=int, default=10)
    parser.add_argument("--horizon-k", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    run_stage7(
        checkpoint_path=args.checkpoint,
        scaler_path=args.scaler,
        state_vectors_dir=args.state_dir,
        output_dir=args.output_dir,
        threat_threshold=args.threat_threshold,
        history_len_m=args.history_m,
        K=args.horizon_k,
        device=args.device
    )
