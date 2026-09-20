"""
Verification Script for Stage 2: Data Preprocessing

This script runs automated sanity checks on the Stage 2 preprocessing pipeline:
1. Column whitespace sanitization.
2. Dropping corrupted/phantom rows.
3. Handling infinities in rate columns.
4. Parsing timestamps into datetime objects and checking strict ascending order.
5. Verifying that attack labels are correctly mapped to canonical classes.
"""

import os
import sys
import numpy as np
import pandas as pd
from data.preprocess import (
    sanitize_column_names,
    drop_corrupted_and_empty_rows,
    clean_infinite_and_null_values,
    parse_and_sort_timestamps,
    standardize_labels,
    LABEL_MAPPING,
    CLASS_TO_INDEX,
)


def run_unit_tests() -> None:
    """Run synthetic unit tests to verify each preprocessing component."""
    print("=" * 60)
    print("Running Stage 2 Preprocessing Unit Tests...")
    print("=" * 60)

    # Test 1: Column whitespace sanitization
    test_df = pd.DataFrame({
        " Destination Port": [80, 443],
        " Flow Duration ": [1000, 2000],
        "Label": ["BENIGN", "DDoS"]
    })
    cleaned_df = sanitize_column_names(test_df)
    assert "Destination Port" in cleaned_df.columns, "Failed to strip column whitespace."
    assert "Flow Duration" in cleaned_df.columns, "Failed to strip column whitespace."
    print("[PASS] Test 1: Column whitespace sanitization works.")

    # Test 2: Dropping corrupted/empty rows
    corrupted_df = pd.DataFrame({
        "Timestamp": ["7/7/2017 1:00", None, "", "7/7/2017 1:02"],
        "Label": ["BENIGN", None, "", "DDoS"],
        "Value": [10.0, np.nan, np.nan, 20.0]
    })
    non_corrupted = drop_corrupted_and_empty_rows(corrupted_df)
    assert len(non_corrupted) == 2, f"Expected 2 valid rows, got {len(non_corrupted)}"
    print("[PASS] Test 2: Corrupted and empty rows dropped cleanly.")

    # Test 3: Handling infinite values
    inf_df = pd.DataFrame({
        "Flow Bytes/s": [100.0, np.inf, 200.0, -np.inf, np.nan],
        "Flow Packets/s": [1.0, 2.0, np.inf, 4.0, 5.0]
    })
    cleaned_inf = clean_infinite_and_null_values(inf_df)
    assert not np.isinf(cleaned_inf["Flow Bytes/s"]).any(), "Infinite values found in Flow Bytes/s."
    assert not np.isinf(cleaned_inf["Flow Packets/s"]).any(), "Infinite values found in Flow Packets/s."
    assert not cleaned_inf.isnull().any().any(), "NaN values found after imputation."
    print("[PASS] Test 3: Infinite values and NaNs safely imputed.")

    # Test 4: Chronological sorting
    unsorted_df = pd.DataFrame({
        "Timestamp": ["7/7/2017 3:00", "7/7/2017 1:00", "7/7/2017 2:00"],
        "Flow Duration": [10, 20, 30]
    })
    sorted_df = parse_and_sort_timestamps(unsorted_df)
    timestamps = sorted_df["Timestamp"].tolist()
    assert timestamps[0] < timestamps[1] < timestamps[2], "Timestamps are not strictly ascending."
    print("[PASS] Test 4: Mixed timestamps parsed and sorted chronologically.")

    # Test 5: Standardizing raw labels into canonical classes
    label_test_df = pd.DataFrame({
        "Label": [
            "BENIGN",
            "FTP-Patator",
            "DoS Hulk",
            "DDoS",
            "PortScan",
            "Web Attack \x96 Brute Force",
            "Bot",
            "Infiltration"
        ]
    })
    labeled_df = standardize_labels(label_test_df)
    expected_categories = [
        "BENIGN", "Brute Force", "DoS", "DDoS",
        "Port Scan", "Web Attack", "Botnet", "Infiltration"
    ]
    assert labeled_df["Attack_Category"].tolist() == expected_categories, "Label mapping mismatch."
    assert labeled_df["Is_Attack"].tolist() == [0, 1, 1, 1, 1, 1, 1, 1], "Binary attack flags mismatch."
    assert labeled_df["Class_Index"].tolist() == list(range(8)), "Class indices mismatch."
    print("[PASS] Test 5: All 8 attack categories correctly mapped.")

    print("\n" + "=" * 60)
    print("All Stage 2 Unit Tests Passed Successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_unit_tests()
