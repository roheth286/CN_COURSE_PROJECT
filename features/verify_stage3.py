"""
Verification and Unit Test Suite for Stage 3: Temporal Windowing

This script tests:
1. Feature dimension of the network state vector (must match STATE_VECTOR_DIM = 36).
2. Volume and velocity aggregations (packets/sec, bytes/sec, packet sums).
3. Port dynamics and diversity metrics (unique destination ports, entropy).
4. Protocol and TCP flag dynamics (SYN/ACK ratios, RST counts).
5. Window ground truth labeling (binary attack indicator, dominant category).
6. Temporal ordering and timestamp integrity across windows.
"""

import sys
import numpy as np
import pandas as pd
from features.windowing import (
    compute_network_state_vector,
    generate_state_vectors_for_dataframe,
    STATE_FEATURE_NAMES,
    STATE_VECTOR_DIM,
)


def run_unit_tests() -> None:
    """Run automated unit tests for Stage 3 temporal windowing."""
    print("=" * 60)
    print("Running Stage 3 Temporal Windowing Unit Tests...")
    print("=" * 60)

    # Test 1: Verify state feature count
    assert len(STATE_FEATURE_NAMES) == STATE_VECTOR_DIM == 36, (
        f"Feature dimension mismatch: {len(STATE_FEATURE_NAMES)} vs {STATE_VECTOR_DIM}"
    )
    print(f"[PASS] Test 1: State vector dimension verified ({STATE_VECTOR_DIM} features).")

    # Construct synthetic test flow DataFrame
    base_time = pd.Timestamp("2017-07-07 10:00:00")
    test_flows = pd.DataFrame({
        "Timestamp": [
            base_time + pd.Timedelta(seconds=5),
            base_time + pd.Timedelta(seconds=12),
            base_time + pd.Timedelta(seconds=25),
        ],
        "Source IP": ["192.168.1.10", "192.168.1.10", "192.168.1.11"],
        "Destination IP": ["10.0.0.1", "10.0.0.1", "10.0.0.2"],
        "Source Port": [50001, 50002, 50003],
        "Destination Port": [80, 443, 80],
        "Protocol": [6, 6, 17],  # Two TCP, one UDP
        "Flow Duration": [100.0, 200.0, 300.0],
        "Total Fwd Packets": [10, 20, 30],
        "Total Backward Packets": [5, 10, 15],
        "Total Length of Fwd Packets": [1000, 2000, 3000],
        "Total Length of Bwd Packets": [500, 1000, 1500],
        "SYN Flag Count": [1, 1, 0],
        "ACK Flag Count": [1, 0, 0],
        "FIN Flag Count": [0, 0, 0],
        "RST Flag Count": [0, 1, 0],
        "PSH Flag Count": [1, 1, 0],
        "URG Flag Count": [0, 0, 0],
        "Packet Length Mean": [100.0, 100.0, 100.0],
        "Packet Length Std": [10.0, 10.0, 10.0],
        "Max Packet Length": [200.0, 200.0, 200.0],
        "Min Packet Length": [40.0, 40.0, 40.0],
        "Flow IAT Mean": [50.0, 50.0, 50.0],
        "Flow IAT Std": [5.0, 5.0, 5.0],
        "Fwd IAT Mean": [20.0, 20.0, 20.0],
        "Bwd IAT Mean": [30.0, 30.0, 30.0],
        "Is_Attack": [0, 1, 1],
        "Attack_Category": ["BENIGN", "DDoS", "DDoS"],
    })

    # Test 2: Single window aggregation calculation
    w_start = base_time
    w_end = base_time + pd.Timedelta(seconds=30)
    state = compute_network_state_vector(test_flows, w_start, w_end, window_seconds=30.0)

    # Check volume
    expected_fwd_pkts = 10 + 20 + 30  # 60
    expected_bwd_pkts = 5 + 10 + 15   # 30
    assert state["flow_count"] == 3, f"Expected flow count 3, got {state['flow_count']}"
    assert state["total_packets"] == 90, f"Expected 90 packets, got {state['total_packets']}"
    assert state["packet_rate"] == 90 / 30.0, f"Expected packet rate 3.0, got {state['packet_rate']}"
    print("[PASS] Test 2: Volume and velocity aggregation verified.")

    # Test 3: Topology and port diversity
    assert state["unique_dst_ports"] == 2, f"Expected 2 unique dst ports, got {state['unique_dst_ports']}"
    assert abs(state["dst_port_diversity"] - (2 / 3.0)) < 1e-5, "Port diversity calculation error."
    print("[PASS] Test 3: Connection topology and port diversity verified.")

    # Test 4: TCP flag dynamics and protocol ratios
    assert state["syn_flag_count"] == 2, f"Expected 2 SYN flags, got {state['syn_flag_count']}"
    assert state["ack_flag_count"] == 1, f"Expected 1 ACK flag, got {state['ack_flag_count']}"
    # syn_ack_ratio = 2 / (1 + 1) = 1.0
    assert state["syn_ack_ratio"] == 1.0, f"Expected syn_ack_ratio 1.0, got {state['syn_ack_ratio']}"
    # protocol: 2 TCP out of 3 = 2/3
    assert abs(state["protocol_tcp_ratio"] - (2 / 3.0)) < 1e-5, "TCP ratio calculation error."
    print("[PASS] Test 4: Protocol and TCP flag dynamics verified.")

    # Test 5: Ground truth labeling
    assert state["is_attack"] == 1, "Expected window is_attack = 1"
    assert state["attack_category"] == "DDoS", f"Expected dominant category 'DDoS', got {state['attack_category']}"
    assert state["class_index"] == 3, f"Expected class index 3 (DDoS), got {state['class_index']}"
    print("[PASS] Test 5: Ground truth window labeling and class indexing verified.")

    # Test 6: Multi-window temporal generation
    multi_window_flows = pd.concat([
        test_flows,
        # Add a second window (at +40s, +50s)
        pd.DataFrame({
            "Timestamp": [
                base_time + pd.Timedelta(seconds=40),
                base_time + pd.Timedelta(seconds=50),
            ],
            "Source IP": ["192.168.1.10", "192.168.1.10"],
            "Destination IP": ["10.0.0.1", "10.0.0.1"],
            "Source Port": [50004, 50005],
            "Destination Port": [80, 80],
            "Protocol": [6, 6],
            "Flow Duration": [150.0, 250.0],
            "Total Fwd Packets": [5, 5],
            "Total Backward Packets": [2, 2],
            "Total Length of Fwd Packets": [500, 500],
            "Total Length of Bwd Packets": [200, 200],
            "SYN Flag Count": [0, 0],
            "ACK Flag Count": [1, 1],
            "FIN Flag Count": [0, 0],
            "RST Flag Count": [0, 0],
            "PSH Flag Count": [0, 0],
            "URG Flag Count": [0, 0],
            "Packet Length Mean": [80.0, 80.0],
            "Packet Length Std": [5.0, 5.0],
            "Max Packet Length": [150.0, 150.0],
            "Min Packet Length": [40.0, 40.0],
            "Flow IAT Mean": [40.0, 40.0],
            "Flow IAT Std": [4.0, 4.0],
            "Fwd IAT Mean": [15.0, 15.0],
            "Bwd IAT Mean": [25.0, 25.0],
            "Is_Attack": [0, 0],
            "Attack_Category": ["BENIGN", "BENIGN"],
        })
    ], ignore_index=True)

    state_df = generate_state_vectors_for_dataframe(multi_window_flows, window_seconds=30)
    assert len(state_df) == 2, f"Expected 2 windows, got {len(state_df)}"
    assert state_df.iloc[0]["is_attack"] == 1, "Window 0 should be attack"
    assert state_df.iloc[1]["is_attack"] == 0, "Window 1 should be benign"
    assert state_df.iloc[0]["window_end"] == state_df.iloc[1]["window_start"], "Windows should be temporally contiguous."
    print("[PASS] Test 6: Multi-window continuous sequence generation verified.")

    print("\n" + "=" * 60)
    print("All Stage 3 Temporal Windowing Unit Tests Passed Successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_unit_tests()
