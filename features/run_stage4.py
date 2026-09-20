"""
Stage 4 Execution Pipeline - Sequence Dataset Construction

This script loads the 30-second state vectors from Datasets/state_vectors/,
performs Per-Session Chronological Splitting (70% Train, 15% Val, 15% Test),
fits a leak-free NetworkStateScaler on the training set, builds sliding-window
sequences (m=10 past windows -> next state S_{t+1} and K=5 future attack targets),
and exports the dataset archives and scaler to Datasets/sequences/.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from features.sequence_dataset import build_full_sequence_dataset
from data.preprocess import INDEX_TO_CLASS


def execute_stage4(
    state_vectors_dir: str = "Datasets/state_vectors",
    output_dir: str = "Datasets/sequences",
    history_len_m: int = 10,
    forecast_horizon_k: int = 5,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> None:
    """Run the complete Stage 4 sequence dataset creation and print verification table."""
    result = build_full_sequence_dataset(
        state_vectors_directory=state_vectors_dir,
        output_directory=output_dir,
        history_len_m=history_len_m,
        forecast_horizon_k=forecast_horizon_k,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
    
    meta = result["metadata"]
    train_x, train_ys, train_yb, train_yc, _ = result["train"]
    val_x, val_ys, val_yb, val_yc, _ = result["val"]
    test_x, test_ys, test_yb, test_yc, _ = result["test"]
    
    print("\n" + "=" * 80)
    print("STAGE 4 SEQUENCE DATASET GENERATION SUMMARY")
    print("=" * 80)
    print(f"Historical Window Length (m)   : {meta['history_len_m']} windows ({meta['history_len_m'] * 30}s = {meta['history_len_m'] * 0.5:.1f} mins)")
    print(f"Forecast Horizon (K)           : {meta['forecast_horizon_k']} windows ({meta['forecast_horizon_k'] * 30}s = {meta['forecast_horizon_k'] * 0.5:.1f} mins lookahead)")
    print(f"State Vector Dimensionality (D): {meta['state_dim']} macroscopic features")
    print("-" * 80)
    
    split_summary = [
        {
            "Split": "Training (70%)",
            "Sequences": len(train_x),
            "Input Shape": str(train_x.shape),
            "Next-State Shape": str(train_ys.shape),
            "Risk Target Shape": str(train_yb.shape),
            "Attack Sequences": int((train_yb[:, 0] == 1).sum()),
            "Attack Rate": f"{(train_yb[:, 0] == 1).mean():.1%}" if len(train_x) > 0 else "0%"
        },
        {
            "Split": "Validation (15%)",
            "Sequences": len(val_x),
            "Input Shape": str(val_x.shape),
            "Next-State Shape": str(val_ys.shape),
            "Risk Target Shape": str(val_yb.shape),
            "Attack Sequences": int((val_yb[:, 0] == 1).sum()),
            "Attack Rate": f"{(val_yb[:, 0] == 1).mean():.1%}" if len(val_x) > 0 else "0%"
        },
        {
            "Split": "Testing (15%)",
            "Sequences": len(test_x),
            "Input Shape": str(test_x.shape),
            "Next-State Shape": str(test_ys.shape),
            "Risk Target Shape": str(test_yb.shape),
            "Attack Sequences": int((test_yb[:, 0] == 1).sum()),
            "Attack Rate": f"{(test_yb[:, 0] == 1).mean():.1%}" if len(test_x) > 0 else "0%"
        },
    ]
    summary_df = pd.DataFrame(split_summary)
    print(summary_df.to_string(index=False))
    print("-" * 80)
    
    # Print class distribution in each split for horizon k=1
    print("\nClass Distribution Across Splits (Immediate Next Window t+1):")
    class_distribution = []
    for c_idx in range(8):
        c_name = INDEX_TO_CLASS.get(c_idx, f"Class {c_idx}")
        n_train = int((train_yc[:, 0] == c_idx).sum()) if len(train_yc) > 0 else 0
        n_val = int((val_yc[:, 0] == c_idx).sum()) if len(val_yc) > 0 else 0
        n_test = int((test_yc[:, 0] == c_idx).sum()) if len(test_yc) > 0 else 0
        class_distribution.append({
            "Class Index": c_idx,
            "Attack Family": c_name,
            "Train Count": n_train,
            "Val Count": n_val,
            "Test Count": n_test,
            "Total Count": n_train + n_val + n_test
        })
    class_df = pd.DataFrame(class_distribution)
    print(class_df.to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4 Sequence Dataset Construction")
    parser.add_argument("--state-vectors-dir", type=str, default="Datasets/state_vectors")
    parser.add_argument("--output-dir", type=str, default="Datasets/sequences")
    parser.add_argument("--history-m", type=int, default=10, help="Number of past windows (m)")
    parser.add_argument("--horizon-k", type=int, default=5, help="Number of future forecast windows (K)")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()
    
    execute_stage4(
        state_vectors_dir=args.state_vectors_dir,
        output_dir=args.output_dir,
        history_len_m=args.history_m,
        forecast_horizon_k=args.horizon_k,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
