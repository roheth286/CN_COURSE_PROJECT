"""
NetForecaster - Temporal Windowing and Network State Vector Construction (Stage 3)

This module converts raw flow-level network traffic into an ordered sequence
of macroscopic "Network State Vectors" (S_t in R^D).

Each state vector aggregates traffic over a fixed temporal window (e.g., 10s, 30s, 60s),
capturing:
1. Volume and velocity (flow counts, packet rates, byte rates).
2. Connection topology (unique destination ports, port diversity/entropy).
3. Protocol and TCP flag dynamics (SYN/ACK ratios, RST counts).
4. Packet geometry and inter-arrival timing (IAT mean and variance).
5. Ground truth labels (is_attack binary indicator, dominant attack category, class index).
"""

import os
import glob
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from data.preprocess import CLASS_TO_INDEX, INDEX_TO_CLASS


# Ordered list of feature names comprising the network state vector S_t
STATE_FEATURE_NAMES: List[str] = [
    # 1. Volume & Velocity
    "flow_count",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_packets",
    "total_fwd_bytes",
    "total_bwd_bytes",
    "total_bytes",
    "packet_rate",
    "byte_rate",
    "fwd_bwd_packet_ratio",
    "fwd_bwd_byte_ratio",
    
    # 2. Connection Topology & Port Dynamics
    "unique_src_ips",
    "unique_dst_ips",
    "unique_src_ports",
    "unique_dst_ports",
    "dst_port_diversity",
    "flow_duration_mean",
    "flow_duration_std",
    
    # 3. Protocol & Flag Dynamics
    "syn_flag_count",
    "ack_flag_count",
    "fin_flag_count",
    "rst_flag_count",
    "psh_flag_count",
    "urg_flag_count",
    "syn_ack_ratio",
    "rst_ack_ratio",
    "protocol_tcp_ratio",
    "protocol_udp_ratio",
    
    # 4. Packet Geometry
    "packet_length_mean",
    "packet_length_std",
    "packet_length_max",
    "packet_length_min",
    
    # 5. Inter-Arrival Timing (IAT)
    "flow_iat_mean",
    "flow_iat_std",
    "fwd_iat_mean",
    "bwd_iat_mean",
]

# Total dimension of the network state vector S_t
STATE_VECTOR_DIM: int = len(STATE_FEATURE_NAMES)  # 36 features


def compute_network_state_vector(
    window_flows: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    window_seconds: float
) -> Dict:
    """
    Compute a single macroscopic Network State Vector S_t from a group of flows.
    
    Parameters:
    -----------
    window_flows : pd.DataFrame
        Flow records occurring strictly within [window_start, window_end).
    window_start : pd.Timestamp
        Start timestamp of the time window.
    window_end : pd.Timestamp
        End timestamp of the time window.
    window_seconds : float
        Duration of the window in seconds.
        
    Returns:
    --------
    Dict containing the state vector features and window ground truth labels.
    """
    flow_count = len(window_flows)
    
    # Volume metrics
    fwd_packets = float(window_flows["Total Fwd Packets"].sum())
    bwd_packets = float(window_flows["Total Backward Packets"].sum())
    total_packets = fwd_packets + bwd_packets
    
    fwd_bytes = float(window_flows["Total Length of Fwd Packets"].sum())
    bwd_bytes = float(window_flows["Total Length of Bwd Packets"].sum())
    total_bytes = fwd_bytes + bwd_bytes
    
    packet_rate = total_packets / window_seconds
    byte_rate = total_bytes / window_seconds
    fwd_bwd_packet_ratio = fwd_packets / (bwd_packets + 1.0)
    fwd_bwd_byte_ratio = fwd_bytes / (bwd_bytes + 1.0)
    
    # Topology and Port Dynamics
    unique_src_ips = float(window_flows["Source IP"].nunique()) if "Source IP" in window_flows.columns else 0.0
    unique_dst_ips = float(window_flows["Destination IP"].nunique()) if "Destination IP" in window_flows.columns else 0.0
    unique_src_ports = float(window_flows["Source Port"].nunique()) if "Source Port" in window_flows.columns else 0.0
    unique_dst_ports = float(window_flows["Destination Port"].nunique()) if "Destination Port" in window_flows.columns else 0.0
    dst_port_diversity = unique_dst_ports / max(flow_count, 1.0)
    
    flow_duration_mean = float(window_flows["Flow Duration"].mean())
    flow_duration_std = float(window_flows["Flow Duration"].std(ddof=0)) if flow_count > 1 else 0.0
    
    # Flag Dynamics
    syn_flags = float(window_flows["SYN Flag Count"].sum()) if "SYN Flag Count" in window_flows.columns else 0.0
    ack_flags = float(window_flows["ACK Flag Count"].sum()) if "ACK Flag Count" in window_flows.columns else 0.0
    fin_flags = float(window_flows["FIN Flag Count"].sum()) if "FIN Flag Count" in window_flows.columns else 0.0
    rst_flags = float(window_flows["RST Flag Count"].sum()) if "RST Flag Count" in window_flows.columns else 0.0
    psh_flags = float(window_flows["PSH Flag Count"].sum()) if "PSH Flag Count" in window_flows.columns else 0.0
    urg_flags = float(window_flows["URG Flag Count"].sum()) if "URG Flag Count" in window_flows.columns else 0.0
    
    syn_ack_ratio = syn_flags / (ack_flags + 1.0)
    rst_ack_ratio = rst_flags / (ack_flags + 1.0)
    
    # Protocol distribution
    if "Protocol" in window_flows.columns:
        tcp_ratio = float((window_flows["Protocol"] == 6).mean())
        udp_ratio = float((window_flows["Protocol"] == 17).mean())
    else:
        tcp_ratio = 1.0
        udp_ratio = 0.0
        
    # Packet Geometry
    pkt_len_mean = float(window_flows["Packet Length Mean"].mean()) if "Packet Length Mean" in window_flows.columns else 0.0
    pkt_len_std = float(window_flows["Packet Length Std"].mean()) if "Packet Length Std" in window_flows.columns else 0.0
    pkt_len_max = float(window_flows["Max Packet Length"].max()) if "Max Packet Length" in window_flows.columns else 0.0
    pkt_len_min = float(window_flows["Min Packet Length"].min()) if "Min Packet Length" in window_flows.columns else 0.0
    
    # Inter-Arrival Times (IAT)
    flow_iat_mean = float(window_flows["Flow IAT Mean"].mean()) if "Flow IAT Mean" in window_flows.columns else 0.0
    flow_iat_std = float(window_flows["Flow IAT Std"].mean()) if "Flow IAT Std" in window_flows.columns else 0.0
    fwd_iat_mean = float(window_flows["Fwd IAT Mean"].mean()) if "Fwd IAT Mean" in window_flows.columns else 0.0
    bwd_iat_mean = float(window_flows["Bwd IAT Mean"].mean()) if "Bwd IAT Mean" in window_flows.columns else 0.0
    
    # Window Ground Truth Labels
    attack_flows = window_flows[window_flows["Is_Attack"] == 1]
    attack_flow_count = len(attack_flows)
    
    if attack_flow_count > 0:
        is_attack = 1
        attack_flow_ratio = attack_flow_count / flow_count
        # Find dominant attack category in this window
        category_counts = attack_flows["Attack_Category"].value_counts()
        dominant_category = category_counts.index[0]
    else:
        is_attack = 0
        attack_flow_ratio = 0.0
        dominant_category = "BENIGN"
        
    class_index = CLASS_TO_INDEX.get(dominant_category, 0)
    
    state_record = {
        # Timestamps
        "window_start": window_start,
        "window_end": window_end,
        
        # Volume & Velocity
        "flow_count": flow_count,
        "total_fwd_packets": fwd_packets,
        "total_bwd_packets": bwd_packets,
        "total_packets": total_packets,
        "total_fwd_bytes": fwd_bytes,
        "total_bwd_bytes": bwd_bytes,
        "total_bytes": total_bytes,
        "packet_rate": packet_rate,
        "byte_rate": byte_rate,
        "fwd_bwd_packet_ratio": fwd_bwd_packet_ratio,
        "fwd_bwd_byte_ratio": fwd_bwd_byte_ratio,
        
        # Connection Topology & Ports
        "unique_src_ips": unique_src_ips,
        "unique_dst_ips": unique_dst_ips,
        "unique_src_ports": unique_src_ports,
        "unique_dst_ports": unique_dst_ports,
        "dst_port_diversity": dst_port_diversity,
        "flow_duration_mean": flow_duration_mean,
        "flow_duration_std": flow_duration_std,
        
        # Protocol & Flags
        "syn_flag_count": syn_flags,
        "ack_flag_count": ack_flags,
        "fin_flag_count": fin_flags,
        "rst_flag_count": rst_flags,
        "psh_flag_count": psh_flags,
        "urg_flag_count": urg_flags,
        "syn_ack_ratio": syn_ack_ratio,
        "rst_ack_ratio": rst_ack_ratio,
        "protocol_tcp_ratio": tcp_ratio,
        "protocol_udp_ratio": udp_ratio,
        
        # Packet Geometry
        "packet_length_mean": pkt_len_mean,
        "packet_length_std": pkt_len_std,
        "packet_length_max": pkt_len_max,
        "packet_length_min": pkt_len_min,
        
        # IAT Timing
        "flow_iat_mean": flow_iat_mean,
        "flow_iat_std": flow_iat_std,
        "fwd_iat_mean": fwd_iat_mean,
        "bwd_iat_mean": bwd_iat_mean,
        
        # Labels & Ground Truth
        "is_attack": is_attack,
        "attack_flow_count": attack_flow_count,
        "attack_flow_ratio": attack_flow_ratio,
        "attack_category": dominant_category,
        "class_index": class_index,
    }
    
    return state_record


def generate_state_vectors_for_dataframe(
    dataframe: pd.DataFrame,
    window_seconds: int = 30
) -> pd.DataFrame:
    """
    Transform a continuous flow dataframe into an ordered sequence of state vectors.
    
    Uses high-speed vectorized interval grouping based on the session's start timestamp.
    
    Parameters:
    -----------
    dataframe : pd.DataFrame
        Cleaned, chronologically sorted network flows from Stage 2.
    window_seconds : int
        Window duration in seconds (default: 30s).
        
    Returns:
    --------
    pd.DataFrame containing the ordered sequence of Network State Vectors.
    """
    if len(dataframe) == 0:
        return pd.DataFrame()
        
    # Ensure dataframe is sorted chronologically
    dataframe = dataframe.sort_values(by="Timestamp", ascending=True).reset_index(drop=True)
    session_start = dataframe["Timestamp"].min()
    
    # Calculate window index for each flow record
    time_deltas = (dataframe["Timestamp"] - session_start).dt.total_seconds()
    window_indices = (time_deltas // window_seconds).astype(int)
    
    dataframe["_window_idx"] = window_indices
    
    state_vector_records = []
    
    # Group flows by window index and compute the state vector
    grouped_windows = dataframe.groupby("_window_idx", sort=True)
    
    for window_idx, window_flows in grouped_windows:
        window_start = session_start + pd.Timedelta(seconds=int(window_idx * window_seconds))
        window_end = window_start + pd.Timedelta(seconds=window_seconds)
        
        state_record = compute_network_state_vector(
            window_flows=window_flows,
            window_start=window_start,
            window_end=window_end,
            window_seconds=float(window_seconds)
        )
        state_vector_records.append(state_record)
        
    # Remove temporary column
    dataframe.drop(columns=["_window_idx"], inplace=True)
    
    state_vectors_df = pd.DataFrame(state_vector_records)
    
    # Fill any NaNs in standard deviation or ratios with 0.0
    for feature_name in STATE_FEATURE_NAMES:
        if feature_name in state_vectors_df.columns:
            state_vectors_df[feature_name] = state_vectors_df[feature_name].fillna(0.0)
            
    return state_vectors_df


def process_parquet_session_to_state_vectors(
    parquet_path: str,
    output_path: str,
    window_seconds: int = 30
) -> pd.DataFrame:
    """
    Load a preprocessed session Parquet file, compute its state vectors, and save to output.
    """
    file_name = os.path.basename(parquet_path)
    print(f"\n[Windowing] Processing: {file_name} (Window = {window_seconds}s)")
    
    df = pd.read_parquet(parquet_path)
    print(f"  [Input] Flows: {len(df):,} | From {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    
    state_df = generate_state_vectors_for_dataframe(df, window_seconds=window_seconds)
    
    print(f"  [Output] Generated {len(state_df):,} temporal state windows.")
    attack_windows = (state_df["is_attack"] == 1).sum()
    benign_windows = len(state_df) - attack_windows
    print(f"  [Window Breakdown] Benign: {benign_windows:,} | Attack: {attack_windows:,}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    state_df.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"  [Saved] Exported state vectors to: {output_path}")
    
    return state_df
