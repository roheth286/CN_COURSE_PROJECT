import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
    print("=" * 60)
    print("Running Stage 2 Preprocessing Unit Tests...")
    print("=" * 60)

    test_df = pd.DataFrame({
        " Destination Port": [80, 443],
        " Flow Duration ": [1000, 2000],
        "Label": ["BENIGN", "DDoS"]
    })
    cleaned_df = sanitize_column_names(test_df)
    assert "Destination Port" in cleaned_df.columns
    assert "Flow Duration" in cleaned_df.columns
    print("[PASS] Test 1: Column whitespace sanitization works.")

    corrupted_df = pd.DataFrame({
        "Timestamp": ["7/7/2017 1:00", None, "", "7/7/2017 1:02"],
        "Label": ["BENIGN", None, "", "DDoS"],
        "Value": [10.0, np.nan, np.nan, 20.0]
    })
    non_corrupted = drop_corrupted_and_empty_rows(corrupted_df)
    assert len(non_corrupted) == 2
    print("[PASS] Test 2: Corrupted and empty rows dropped cleanly.")

    inf_df = pd.DataFrame({
        "Flow Bytes/s": [100.0, np.inf, 200.0, -np.inf, np.nan],
        "Flow Packets/s": [1.0, 2.0, np.inf, 4.0, 5.0]
    })
    cleaned_inf = clean_infinite_and_null_values(inf_df)
    assert not np.isinf(cleaned_inf["Flow Bytes/s"]).any()
    assert not np.isinf(cleaned_inf["Flow Packets/s"]).any()
    assert not cleaned_inf.isnull().any().any()
    print("[PASS] Test 3: Infinite values and NaNs safely imputed.")

    unsorted_df = pd.DataFrame({
        "Timestamp": ["7/7/2017 3:00", "7/7/2017 1:00", "7/7/2017 2:00"],
        "Flow Duration": [10, 20, 30]
    })
    sorted_df = parse_and_sort_timestamps(unsorted_df)
    timestamps = sorted_df["Timestamp"].tolist()
    assert timestamps[0] < timestamps[1] < timestamps[2]
    print("[PASS] Test 4: Mixed timestamps parsed and sorted chronologically.")

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
    assert labeled_df["Attack_Category"].tolist() == expected_categories
    assert labeled_df["Is_Attack"].tolist() == [0, 1, 1, 1, 1, 1, 1, 1]
    assert labeled_df["Class_Index"].tolist() == list(range(8))
    print("[PASS] Test 5: All 8 attack categories correctly mapped.")

    print("\n" + "=" * 60)
    print("All Stage 2 Unit Tests Passed Successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_unit_tests()
