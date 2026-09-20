"""
Stage 3 Execution Pipeline - Batch Temporal Windowing

This script processes all cleaned Parquet files from Datasets/processed/,
aggregates flows into macroscopic Network State Vectors (S_t), and exports
them as structured temporal time series to Datasets/state_vectors/.
"""

import os
import sys
import glob
import argparse
import pandas as pd
from typing import List, Dict
from features.windowing import process_parquet_session_to_state_vectors, STATE_VECTOR_DIM


def batch_window_all_sessions(
    input_directory: str = "Datasets/processed",
    output_directory: str = "Datasets/state_vectors",
    window_seconds: int = 30,
    max_files: int = 0
) -> List[Dict]:
    """
    Process all session Parquet files into temporal network state vectors.
    
    Parameters:
    -----------
    input_directory : str
        Directory containing the preprocessed session Parquet files.
    output_directory : str
        Directory where generated state vector Parquet files will be stored.
    window_seconds : int
        Duration of each aggregation window in seconds (default: 30s).
    max_files : int
        If > 0, process only the first N files.
    """
    os.makedirs(output_directory, exist_ok=True)
    
    parquet_pattern = os.path.join(input_directory, "*.parquet")
    parquet_files = sorted(glob.glob(parquet_pattern))
    
    if not parquet_files:
        raise FileNotFoundError(f"No preprocessed Parquet files found in: {input_directory}")
        
    if max_files > 0:
        parquet_files = parquet_files[:max_files]
        print(f"[Notice] Processing limited to first {max_files} file(s).")
        
    print("=" * 75)
    print(f"Starting Stage 3 Temporal Windowing (Window = {window_seconds}s, Files = {len(parquet_files)})")
    print(f"State Vector Dimensionality: {STATE_VECTOR_DIM} features")
    print("=" * 75)
    
    summary_records = []
    
    for file_index, file_path in enumerate(parquet_files, start=1):
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        output_file_path = os.path.join(
            output_directory,
            f"{base_name}_window_{window_seconds}s.parquet"
        )
        
        print(f"\n[{file_index}/{len(parquet_files)}] Windowing session: {file_name}")
        
        # Execute Stage 3 temporal windowing
        state_df = process_parquet_session_to_state_vectors(
            parquet_path=file_path,
            output_path=output_file_path,
            window_seconds=window_seconds
        )
        
        total_windows = len(state_df)
        attack_windows = int((state_df["is_attack"] == 1).sum()) if total_windows > 0 else 0
        benign_windows = total_windows - attack_windows
        
        # Categorical breakdown
        cat_counts = dict(state_df["attack_category"].value_counts()) if total_windows > 0 else {}
        attack_types_str = ", ".join(
            f"{cat}({cnt})" for cat, cnt in cat_counts.items() if cat != "BENIGN"
        ) or "None (Benign)"
        
        summary_records.append({
            "Session": file_name,
            "Total_Windows": total_windows,
            "Benign_Windows": benign_windows,
            "Attack_Windows": attack_windows,
            "Attack_Types": attack_types_str
        })
        
    # Print final summary table
    print("\n" + "=" * 80)
    print(f"STAGE 3 TEMPORAL WINDOWING SUMMARY (Window Size = {window_seconds}s)")
    print("=" * 80)
    summary_df = pd.DataFrame(summary_records)
    print(summary_df.to_string(index=False))
    print("=" * 80 + "\n")
    
    return summary_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3 Temporal Windowing for NetForecaster")
    parser.add_argument("--input-dir", type=str, default="Datasets/processed", help="Path to input Parquet directory")
    parser.add_argument("--output-dir", type=str, default="Datasets/state_vectors", help="Path to output state vectors directory")
    parser.add_argument("--window-size", type=int, default=30, help="Window duration in seconds (default: 30)")
    parser.add_argument("--max-files", type=int, default=0, help="Maximum number of files to process (0 for all)")
    args = parser.parse_args()
    
    batch_window_all_sessions(
        input_directory=args.input_dir,
        output_directory=args.output_dir,
        window_seconds=args.window_size,
        max_files=args.max_files
    )
