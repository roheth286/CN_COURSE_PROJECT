"""
Stage 2 Execution Pipeline - Batch Preprocessing

This script processes all raw CIC-IDS2017 session files from Datasets/TrafficLabelling/,
cleans and normalizes them, and saves the cleaned datasets as high-performance
Parquet files in data/processed/.
"""

import os
import sys
import glob
import argparse
import pandas as pd
from typing import List, Dict
from data.preprocess import load_and_preprocess_single_file


def process_all_sessions(
    input_directory: str = "Datasets/TrafficLabelling",
    output_directory: str = "Datasets/processed",
    max_files: int = 0
) -> List[Dict]:
    """
    Load, clean, and export all network session CSVs in chronological order.
    
    Parameters:
    -----------
    input_directory : str
        Directory containing the raw CIC-IDS2017 CSV files.
    output_directory : str
        Directory where preprocessed Parquet files will be saved.
    max_files : int
        If > 0, process only the first N files (useful for quick testing).
    """
    os.makedirs(output_directory, exist_ok=True)
    
    csv_pattern = os.path.join(input_directory, "*.csv")
    csv_files = sorted(glob.glob(csv_pattern))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {input_directory}")
        
    if max_files > 0:
        csv_files = csv_files[:max_files]
        print(f"[Notice] Processing limited to first {max_files} file(s).")
        
    print("=" * 70)
    print(f"Starting Stage 2 Batch Preprocessing ({len(csv_files)} files)")
    print("=" * 70)
    
    summary_records = []
    
    for file_index, file_path in enumerate(csv_files, start=1):
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        output_file_path = os.path.join(output_directory, f"{base_name}.parquet")
        
        print(f"\n[{file_index}/{len(csv_files)}] Processing: {file_name}")
        
        # Execute Stage 2 preprocessing pipeline
        cleaned_df = load_and_preprocess_single_file(file_path)
        
        # Save as Parquet for speed, compression, and exact type preservation
        print(f"  [Saving] Exporting to: {output_file_path}")
        cleaned_df.to_parquet(output_file_path, index=False, engine="pyarrow")
        
        # Collect summary statistics
        attack_counts = dict(cleaned_df["Attack_Category"].value_counts())
        total_attacks = sum(count for cat, count in attack_counts.items() if cat != "BENIGN")
        
        summary_records.append({
            "File": file_name,
            "Cleaned_Rows": len(cleaned_df),
            "Start_Time": str(cleaned_df["Timestamp"].min()),
            "End_Time": str(cleaned_df["Timestamp"].max()),
            "Total_Attacks": total_attacks,
            "Attack_Types": ", ".join(f"{cat}({cnt:,})" for cat, cnt in attack_counts.items() if cat != "BENIGN") or "None (Benign)"
        })
        
    # Print final summary table
    print("\n" + "=" * 80)
    print("STAGE 2 PREPROCESSING SUMMARY TABLE")
    print("=" * 80)
    summary_df = pd.DataFrame(summary_records)
    print(summary_df.to_string(index=False))
    print("=" * 80 + "\n")
    
    return summary_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 Preprocessing for NetForecaster")
    parser.add_argument("--input-dir", type=str, default="Datasets/TrafficLabelling", help="Path to raw CSV directory")
    parser.add_argument("--output-dir", type=str, default="Datasets/processed", help="Path to output Parquet directory")
    parser.add_argument("--max-files", type=int, default=0, help="Maximum number of files to process (0 for all)")
    args = parser.parse_args()
    
    process_all_sessions(
        input_directory=args.input_dir,
        output_directory=args.output_dir,
        max_files=args.max_files
    )
