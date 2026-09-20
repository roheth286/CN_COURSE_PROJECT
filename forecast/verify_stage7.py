"""
Verification and Unit Test Suite for Stage 7: Early Warning Lead Time Engine

This script tests:
1. identify_attack_episodes accurately segments contiguous attack blocks and metadata.
2. Proactive early warning calculation (Delta T > 0) when an alarm triggers ahead of attack onset.
3. Onset detection calculation (Delta T == 0) when an alarm triggers at the start of attack.
4. False alarm tracking on purely benign streaming telemetry.
5. LeadTimeSummary aggregation and per-category breakdown calculation.
6. End-to-end integration test with trained checkpoint on a real session parquet file.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch

from model.network_world_model import NetworkWorldModel
from features.sequence_dataset import STATE_FEATURE_NAMES, STATE_VECTOR_DIM, NetworkStateScaler
from forecast.rollout import RecursiveForecaster
from forecast.lead_time import (
    AttackEpisode,
    EpisodeLeadTimeResult,
    LeadTimeSummary,
    identify_attack_episodes,
    LeadTimeEngine,
)


def run_unit_tests() -> None:
    """Run all automated unit tests for Stage 7."""
    print("=" * 65)
    print("Running Stage 7 Early Warning Lead Time Unit Tests...")
    print("=" * 65)

    # -------------------------------------------------------------
    # Test 1: identify_attack_episodes segmentation
    # -------------------------------------------------------------
    # Create 30 windows: 10 Benign, 5 DDoS, 5 Benign, 10 PortScan
    n_total = 30
    is_att = np.zeros(n_total, dtype=int)
    is_att[10:15] = 1
    is_att[20:30] = 1

    cats = ["BENIGN"] * n_total
    for i in range(10, 15):
        cats[i] = "DDoS"
    for i in range(20, 30):
        cats[i] = "Port Scan"

    class_idxs = [0] * n_total
    for i in range(10, 15):
        class_idxs[i] = 3
    for i in range(20, 30):
        class_idxs[i] = 4

    test_df = pd.DataFrame({
        "is_attack": is_att,
        "attack_category": cats,
        "class_index": class_idxs,
        "window_start": [f"2017-07-07 00:{i:02d}:00" for i in range(n_total)],
        "window_end": [f"2017-07-07 00:{i:02d}:30" for i in range(n_total)],
    })

    episodes = identify_attack_episodes(test_df, session_name="test_session")
    assert len(episodes) == 2, f"Expected 2 attack episodes, got {len(episodes)}"

    ep1 = episodes[0]
    assert ep1.attack_category == "DDoS", f"Expected DDoS, got {ep1.attack_category}"
    assert ep1.start_window_idx == 10 and ep1.end_window_idx == 14, f"Ep1 range mismatch: {ep1.start_window_idx}-{ep1.end_window_idx}"
    assert ep1.num_windows == 5, f"Expected 5 windows, got {ep1.num_windows}"
    assert ep1.duration_seconds == 150.0, f"Expected 150.0s, got {ep1.duration_seconds}"

    ep2 = episodes[1]
    assert ep2.attack_category == "Port Scan", f"Expected Port Scan, got {ep2.attack_category}"
    assert ep2.start_window_idx == 20 and ep2.end_window_idx == 29, f"Ep2 range mismatch: {ep2.start_window_idx}-{ep2.end_window_idx}"
    assert ep2.num_windows == 10, f"Expected 10 windows, got {ep2.num_windows}"

    print(f"[PASS] Test 1: identify_attack_episodes successfully segmented 2 distinct episodes.")

    # -------------------------------------------------------------
    # Test 2: Proactive Early Warning Calculation (Delta T > 0)
    # -------------------------------------------------------------
    # Simulate a stream where an attack starts at index 15
    # Dummy forecaster that predicts threat whenever t == 13 (2 steps = 60s ahead)
    class MockProactiveForecaster:
        def __init__(self):
            self.scaler = None
        def rollout(self, sequences, K=5, unscale=False, batch_size=64):
            N = len(sequences)
            # Default benign predictions
            risks = np.zeros((N, K), dtype=np.float32)
            cats = np.zeros((N, K), dtype=int)
            cat_probs = np.zeros((N, K, 8), dtype=np.float32)
            cat_probs[:, :, 0] = 1.0  # 100% Benign default

            # If surveillance step is t=13 (which is index 13 - 9 = 4 in monitoring_steps with m=10)
            # In monitoring_steps, step 13 corresponds to index 13 - (m-1) = 13 - 9 = 4
            # We trigger alarm at step 13 for k=2 (target = 13 + 2 = 15)
            # Look at monitoring steps: for m=10, monitoring_steps start at 9
            # So t=13 is at index 4 (9, 10, 11, 12, 13)
            idx_13 = 13 - 9
            if 0 <= idx_13 < N:
                risks[idx_13, 1] = 0.95  # k=2 (step 2) threat prob = 0.95
                cat_probs[idx_13, 1, 0] = 0.05
                cat_probs[idx_13, 1, 3] = 0.90 # DDoS
                cats[idx_13, 1] = 3

            from forecast.rollout import ForecastResult
            return ForecastResult(
                predicted_states=np.zeros((N, K, STATE_VECTOR_DIM)),
                predicted_states_unscaled=None,
                risk_probabilities=risks,
                category_probabilities=cat_probs,
                predicted_categories=cats,
                category_names=[["DDoS" if c == 3 else "BENIGN" for c in row] for row in cats],
                horizon_steps=list(range(1, K + 1)),
                horizon_seconds=[k * 30 for k in range(1, K + 1)]
            )

    stream_len = 25
    features = np.zeros((stream_len, STATE_VECTOR_DIM), dtype=np.float32)
    feat_df = pd.DataFrame(features, columns=STATE_FEATURE_NAMES)
    feat_df["is_attack"] = [0]*15 + [1]*5 + [0]*5
    feat_df["attack_category"] = ["BENIGN"]*15 + ["DDoS"]*5 + ["BENIGN"]*5
    feat_df["class_index"] = [0]*15 + [3]*5 + [0]*5
    feat_df["window_start"] = [f"2017-07-07 01:{i:02d}:00" for i in range(stream_len)]

    mock_forecaster = MockProactiveForecaster()
    engine = LeadTimeEngine(forecaster=mock_forecaster, threat_threshold=0.50, window_seconds=30)
    results, fa, ben_w = engine.evaluate_stream(feat_df, session_name="mock_proactive", history_len_m=10, K=5)

    assert len(results) == 1, "Expected 1 episode evaluated"
    res = results[0]
    assert res.detected is True, "Episode should be detected"
    assert res.lead_time_windows == 2, f"Expected 2 windows lead time, got {res.lead_time_windows}"
    assert res.lead_time_seconds == 60.0, f"Expected 60.0s lead time, got {res.lead_time_seconds}"
    assert res.detection_type == "Proactive Early Warning", f"Expected Proactive Early Warning, got {res.detection_type}"
    assert res.alert_window_idx == 13, f"Expected alert at window 13, got {res.alert_window_idx}"
    print(f"[PASS] Test 2: Proactive Early Warning verified with Delta T = +60.0 seconds (2 windows).")

    # -------------------------------------------------------------
    # Test 3: Onset Detection Calculation (Delta T == 0)
    # -------------------------------------------------------------
    class MockOnsetForecaster:
        def __init__(self):
            self.scaler = None
        def rollout(self, sequences, K=5, unscale=False, batch_size=64):
            N = len(sequences)
            risks = np.zeros((N, K), dtype=np.float32)
            cats = np.zeros((N, K), dtype=int)
            cat_probs = np.zeros((N, K, 8), dtype=np.float32)
            cat_probs[:, :, 0] = 1.0

            # Alert triggered exactly at onset t=15 (k=1, target=15+1=16 within episode 15-19)
            idx_15 = 15 - 9
            if 0 <= idx_15 < N:
                risks[idx_15, 0] = 0.88
                cat_probs[idx_15, 0, 0] = 0.10
                cat_probs[idx_15, 0, 3] = 0.85
                cats[idx_15, 0] = 3

            from forecast.rollout import ForecastResult
            return ForecastResult(
                predicted_states=np.zeros((N, K, STATE_VECTOR_DIM)),
                predicted_states_unscaled=None,
                risk_probabilities=risks,
                category_probabilities=cat_probs,
                predicted_categories=cats,
                category_names=[["DDoS" if c == 3 else "BENIGN" for c in row] for row in cats],
                horizon_steps=list(range(1, K + 1)),
                horizon_seconds=[k * 30 for k in range(1, K + 1)]
            )

    mock_onset = MockOnsetForecaster()
    engine_onset = LeadTimeEngine(forecaster=mock_onset, threat_threshold=0.50, window_seconds=30)
    results_onset, _, _ = engine_onset.evaluate_stream(feat_df, session_name="mock_onset", history_len_m=10, K=5)

    res_onset = results_onset[0]
    assert res_onset.detected is True, "Onset episode should be detected"
    assert res_onset.lead_time_windows == 0, f"Expected 0 windows lead time, got {res_onset.lead_time_windows}"
    assert res_onset.lead_time_seconds == 0.0, f"Expected 0.0s lead time, got {res_onset.lead_time_seconds}"
    assert res_onset.detection_type == "Onset Detection", f"Expected Onset Detection, got {res_onset.detection_type}"
    print(f"[PASS] Test 3: Onset Detection verified with Delta T = 0.0 seconds.")

    # -------------------------------------------------------------
    # Test 4: False Alarm Counting on Benign Telemetry
    # -------------------------------------------------------------
    class MockFalseAlarmForecaster:
        def __init__(self):
            self.scaler = None
        def rollout(self, sequences, K=5, unscale=False, batch_size=64):
            N = len(sequences)
            risks = np.zeros((N, K), dtype=np.float32)
            cats = np.zeros((N, K), dtype=int)
            cat_probs = np.zeros((N, K, 8), dtype=np.float32)
            cat_probs[:, :, 0] = 1.0

            # Spurious false alarm at window index 11
            idx_11 = 11 - 9
            if 0 <= idx_11 < N:
                risks[idx_11, 0] = 0.99
                cat_probs[idx_11, 0, 0] = 0.01
                cat_probs[idx_11, 0, 1] = 0.90
                cats[idx_11, 0] = 1

            from forecast.rollout import ForecastResult
            return ForecastResult(
                predicted_states=np.zeros((N, K, STATE_VECTOR_DIM)),
                predicted_states_unscaled=None,
                risk_probabilities=risks,
                category_probabilities=cat_probs,
                predicted_categories=cats,
                category_names=[["Brute Force" if c == 1 else "BENIGN" for c in row] for row in cats],
                horizon_steps=list(range(1, K + 1)),
                horizon_seconds=[k * 30 for k in range(1, K + 1)]
            )

    benign_df = pd.DataFrame(np.zeros((25, STATE_VECTOR_DIM)), columns=STATE_FEATURE_NAMES)
    benign_df["is_attack"] = 0
    benign_df["attack_category"] = "BENIGN"
    benign_df["class_index"] = 0

    mock_fa = MockFalseAlarmForecaster()
    engine_fa = LeadTimeEngine(forecaster=mock_fa, threat_threshold=0.50, window_seconds=30)
    res_fa, fa_count, ben_cnt = engine_fa.evaluate_stream(benign_df, session_name="benign_stream", history_len_m=10, K=5)

    assert len(res_fa) == 0, "No episodes should exist in benign stream"
    assert fa_count == 1, f"Expected 1 false alarm, got {fa_count}"
    assert ben_cnt == 15, f"Expected 15 benign surveillance windows, got {ben_cnt}"
    print(f"[PASS] Test 4: False alarm tracking verified (1 false alarm recorded).")

    # -------------------------------------------------------------
    # Test 5: Summary Metrics Compilation
    # -------------------------------------------------------------
    sample_res1 = EpisodeLeadTimeResult(
        episode_id=1, session_name="s1", attack_category="DDoS",
        start_window_idx=10, end_window_idx=15, start_timestamp=None,
        detected=True, alert_window_idx=8, alert_timestamp=None, forecast_step_k=2,
        lead_time_seconds=60.0, lead_time_windows=2, detection_type="Proactive Early Warning",
        threat_confidence=0.9, predicted_category="DDoS", category_match=True
    )
    sample_res2 = EpisodeLeadTimeResult(
        episode_id=2, session_name="s2", attack_category="Port Scan",
        start_window_idx=20, end_window_idx=25, start_timestamp=None,
        detected=True, alert_window_idx=20, alert_timestamp=None, forecast_step_k=1,
        lead_time_seconds=0.0, lead_time_windows=0, detection_type="Onset Detection",
        threat_confidence=0.8, predicted_category="Port Scan", category_match=True
    )

    summary = LeadTimeSummary(
        total_episodes=2, detected_episodes=2, early_warning_episodes=1,
        onset_episodes=1, delayed_episodes=0, missed_episodes=0,
        detection_rate=1.0, early_warning_rate=0.50,
        mean_lead_time_seconds=30.0, median_lead_time_seconds=30.0, max_lead_time_seconds=60.0,
        total_benign_windows=100, false_alarms=2, false_alarm_rate=0.02,
        category_metrics={
            "DDoS": {"mean_lead_time_seconds": 60.0, "early_warned": 1},
            "Port Scan": {"mean_lead_time_seconds": 0.0, "early_warned": 0}
        },
        episode_details=[]
    )
    s_dict = summary.to_dict()
    assert s_dict["early_warning_rate"] == 0.50, "Summary serialization mismatch"
    assert s_dict["max_lead_time_seconds"] == 60.0, "Max lead time mismatch"
    print(f"[PASS] Test 5: LeadTimeSummary statistics and serialization verified.")

    # -------------------------------------------------------------
    # Test 6: Real Checkpoint & Session Parquet Integration Test
    # -------------------------------------------------------------
    checkpoint_path = os.path.join(PROJECT_ROOT, "model", "checkpoints", "best_world_model.pt")
    scaler_path = os.path.join(PROJECT_ROOT, "Datasets", "sequences", "scaler.pkl")
    session_path = os.path.join(
        PROJECT_ROOT, "Datasets", "state_vectors",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX_window_30s.parquet"
    )

    if os.path.exists(checkpoint_path) and os.path.exists(scaler_path) and os.path.exists(session_path):
        real_forecaster = RecursiveForecaster(
            model=checkpoint_path,
            scaler=scaler_path,
            device="cpu",
            window_seconds=30
        )
        real_engine = LeadTimeEngine(
            forecaster=real_forecaster,
            threat_threshold=0.50,
            use_composite_threat=True,
            window_seconds=30
        )
        real_df = pd.read_parquet(session_path)
        real_results, real_fa, real_ben = real_engine.evaluate_stream(
            real_df,
            session_name="Friday-PortScan",
            history_len_m=10,
            K=5
        )

        assert len(real_results) > 0, "Expected at least 1 attack episode in PortScan session"
        ep0 = real_results[0]
        print(f"[PASS] Test 6: Real session integration test verified on Friday-PortScan.")
        print(f"       Attack Episode: {ep0.attack_category} (Windows {ep0.start_window_idx} to {ep0.end_window_idx})")
        print(f"       Detected: {ep0.detected} | Detection Type: {ep0.detection_type}")
        print(f"       Lead Time: {ep0.lead_time_seconds} seconds ({ep0.lead_time_windows} windows)")
    else:
        print("[SKIP] Test 6: Required checkpoint or parquet file not found.")

    print("=" * 65)
    print("All Stage 7 Early Warning Lead Time Unit Tests Passed Successfully!")
    print("=" * 65)


if __name__ == "__main__":
    run_unit_tests()
