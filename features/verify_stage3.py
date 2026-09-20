import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from features.windowing import (
    compute_network_state_vector,
    generate_state_vectors_for_dataframe,
    STATE_FEATURE_NAMES,
    STATE_VECTOR_DIM,
)


def run_unit_tests() -> None:
    print("=" * 60)
    print("Running Stage 3 Temporal Windowing Unit Tests...")
    print("=" * 60)

    assert len(STATE_FEATURE_NAMES) == STATE_VECTOR_DIM == 36
    print(f"[PASS] Test 1: State vector dimension verified ({STATE_VECTOR_DIM} features).")

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
        "Protocol": [6, 6, 17],
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

    w_start = base_time
    w_end = base_time + pd.Timedelta(seconds=30)
    state = compute_network_state_vector(test_flows, w_start, w_end, window_seconds=30.0)

    assert state["flow_count"] == 3
    assert state["total_packets"] == 90
    assert state["packet_rate"] == 90 / 30.0
    print("[PASS] Test 2: Volume and velocity aggregation verified.")

    assert state["unique_dst_ports"] == 2
    assert abs(state["dst_port_diversity"] - (2 / 3.0)) < 1e-5
    print("[PASS] Test 3: Connection topology and port diversity verified.")

    assert state["syn_flag_count"] == 2
    assert state["ack_flag_count"] == 1
    assert state["syn_ack_ratio"] == 1.0
    assert abs(state["protocol_tcp_ratio"] - (2 / 3.0)) < 1e-5
    print("[PASS] Test 4: Protocol and TCP flag dynamics verified.")

    assert state["is_attack"] == 1
    assert state["attack_category"] == "DDoS"
    assert state["class_index"] == 3
    print("[PASS] Test 5: Ground truth window labeling and class indexing verified.")

    multi_window_flows = pd.concat([
        test_flows,
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
    assert len(state_df) == 2
    assert state_df.iloc[0]["is_attack"] == 1
    assert state_df.iloc[1]["is_attack"] == 0
    assert state_df.iloc[0]["window_end"] == state_df.iloc[1]["window_start"]
    print("[PASS] Test 6: Multi-window continuous sequence generation verified.")

    print("\n" + "=" * 60)
    print("All Stage 3 Temporal Windowing Unit Tests Passed Successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_unit_tests()
