"""
NetForecaster - Data Preprocessing Module (Stage 2)

This module handles:
1. Column name sanitization (stripping whitespace).
2. Dropping corrupted/empty rows (such as the 288,602 blank lines in Thursday morning).
3. Handling missing values and infinite numbers in network rate metrics.
4. Parsing timestamps into datetime objects and sorting chronologically.
5. Standardizing raw attack strings into 8 canonical classes.
6. Generating binary attack indicators (0 = Benign, 1 = Attack).
"""

import os
import glob
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional


# Canonical 8-class mapping for CIC-IDS2017
LABEL_MAPPING: Dict[str, str] = {
    # Benign traffic
    "BENIGN": "BENIGN",
    
    # Brute Force attacks (Tuesday)
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
    
    # Denial of Service (Wednesday)
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "Heartbleed": "DoS",
    
    # Distributed Denial of Service (Friday Afternoon)
    "DDoS": "DDoS",
    
    # Reconnaissance / Scanning (Friday Afternoon)
    "PortScan": "Port Scan",
    
    # Web Application Attacks (Thursday Morning)
    "Web Attack - Brute Force": "Web Attack",
    "Web Attack \x96 Brute Force": "Web Attack",
    "Web Attack - XSS": "Web Attack",
    "Web Attack \x96 XSS": "Web Attack",
    "Web Attack - Sql Injection": "Web Attack",
    "Web Attack \x96 Sql Injection": "Web Attack",
    
    # Botnet Activity (Friday Morning)
    "Bot": "Botnet",
    
    # Infiltration / Targeted Exploit (Thursday Afternoon)
    "Infiltration": "Infiltration",
}

# Integer class index mapping for the 8 canonical classes
CLASS_TO_INDEX: Dict[str, int] = {
    "BENIGN": 0,
    "Brute Force": 1,
    "DoS": 2,
    "DDoS": 3,
    "Port Scan": 4,
    "Web Attack": 5,
    "Botnet": 6,
    "Infiltration": 7,
}

INDEX_TO_CLASS: Dict[int, str] = {
    index: label_name for label_name, index in CLASS_TO_INDEX.items()
}


def sanitize_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Remove leading and trailing whitespace from all column names.
    
    Many CIC-IDS2017 columns have accidental spaces, for example
    ' Destination Port' instead of 'Destination Port'.
    """
    cleaned_column_names = []
    for column_name in dataframe.columns:
        stripped_name = column_name.strip()
        cleaned_column_names.append(stripped_name)
    
    dataframe.columns = cleaned_column_names
    return dataframe


def drop_corrupted_and_empty_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows that are entirely empty or missing critical identifiers.
    
    Thursday-WorkingHours-Morning has 288,602 trailing empty comma rows
    that must be dropped to prevent data corruption.
    """
    initial_row_count = len(dataframe)
    
    # Drop rows where all elements or the core identifiers are null
    dataframe = dataframe.dropna(how="all")
    
    # Check if Timestamp and Label columns exist, drop if both are null
    if "Timestamp" in dataframe.columns:
        dataframe = dataframe.dropna(subset=["Timestamp"])
        
    if "Label" in dataframe.columns:
        dataframe = dataframe.dropna(subset=["Label"])
        # Also drop rows where label is just an empty string
        dataframe = dataframe[dataframe["Label"].astype(str).str.strip() != ""]
        
    final_row_count = len(dataframe)
    dropped_count = initial_row_count - final_row_count
    if dropped_count > 0:
        print(f"  [Clean] Dropped {dropped_count:,} empty/corrupted rows.")
        
    return dataframe


def clean_infinite_and_null_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Replace infinite numbers with NaN and handle missing values.
    
    When Flow Duration is 0, columns like 'Flow Bytes/s' and 'Flow Packets/s'
    contain positive or negative Infinity. We replace them with column max/medians.
    """
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    
    for column_name in numeric_columns:
        column_series = dataframe[column_name]
        
        # Check if there are infinite values
        has_infinite = np.isinf(column_series).any()
        if has_infinite:
            # Replace +inf and -inf with NaN first
            dataframe[column_name] = dataframe[column_name].replace([np.inf, -np.inf], np.nan)
            
            # Find the 99.9th percentile to cap extreme runaway values safely
            valid_numbers = dataframe[column_name].dropna()
            if len(valid_numbers) > 0:
                cap_value = valid_numbers.quantile(0.999)
                dataframe[column_name] = dataframe[column_name].fillna(cap_value)
            else:
                dataframe[column_name] = dataframe[column_name].fillna(0.0)
                
        # Fill any remaining NaNs with column median
        if dataframe[column_name].isnull().any():
            median_value = dataframe[column_name].median()
            dataframe[column_name] = dataframe[column_name].fillna(median_value)
            
    return dataframe


def parse_and_sort_timestamps(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the mixed timestamp formats into standard datetime objects and sort ascending.
    
    Chronological sorting is mandatory because NetForecaster is a temporal forecasting system.
    """
    if "Timestamp" not in dataframe.columns:
        raise ValueError("The dataframe does not contain a 'Timestamp' column.")
        
    # Convert to standard datetime using flexible dayfirst parsing
    dataframe["Timestamp"] = pd.to_datetime(
        dataframe["Timestamp"].astype(str).str.strip(),
        format="mixed",
        dayfirst=True,
        errors="coerce"
    )
    
    # Drop any rows where timestamp could not be parsed
    dataframe = dataframe.dropna(subset=["Timestamp"])
    
    # Sort strictly by timestamp ascending
    dataframe = dataframe.sort_values(by="Timestamp", ascending=True)
    dataframe = dataframe.reset_index(drop=True)
    
    return dataframe


def standardize_labels(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Map raw dataset labels to the 8 canonical classes and create a binary indicator.
    
    Adds two new standardized columns:
    - 'Attack_Category': One of the 8 canonical class strings.
    - 'Is_Attack': 0 for BENIGN, 1 for any malicious traffic.
    - 'Class_Index': Integer from 0 to 7.
    """
    if "Label" not in dataframe.columns:
        raise ValueError("The dataframe does not contain a 'Label' column.")
        
    raw_labels = dataframe["Label"].astype(str).str.strip()
    
    standardized_categories = []
    is_attack_flags = []
    class_indices = []
    
    for raw_label in raw_labels:
        # Match using the canonical mapping dictionary
        matched_category = LABEL_MAPPING.get(raw_label, None)
        
        if matched_category is None:
            # Fallback check for web attacks with varying dash encodings
            if "web attack" in raw_label.lower():
                matched_category = "Web Attack"
            elif "benign" in raw_label.lower():
                matched_category = "BENIGN"
            else:
                matched_category = "BENIGN"
                
        standardized_categories.append(matched_category)
        
        # Binary flag: 0 if BENIGN, 1 if malicious
        if matched_category == "BENIGN":
            is_attack_flags.append(0)
        else:
            is_attack_flags.append(1)
            
        # Integer class index (0 to 7)
        class_index = CLASS_TO_INDEX.get(matched_category, 0)
        class_indices.append(class_index)
        
    dataframe["Attack_Category"] = standardized_categories
    dataframe["Is_Attack"] = is_attack_flags
    dataframe["Class_Index"] = class_indices
    
    return dataframe


def load_and_preprocess_single_file(file_path: str) -> pd.DataFrame:
    """
    Load and execute the complete Stage 2 preprocessing pipeline on a single CSV file.
    """
    file_name = os.path.basename(file_path)
    print(f"\n[Processing] Loading file: {file_name}")
    
    # Read CSV with latin-1 to safely handle non-ascii characters in labels
    dataframe = pd.read_csv(
        file_path,
        encoding="latin-1",
        low_memory=False
    )
    print(f"  [Loaded] Raw rows: {len(dataframe):,} | Columns: {len(dataframe.columns)}")
    
    # Step 1: Sanitize column headers
    dataframe = sanitize_column_names(dataframe)
    
    # Step 2: Drop corrupted/empty trailing rows
    dataframe = drop_corrupted_and_empty_rows(dataframe)
    
    # Step 3: Handle infinite and null numeric values
    dataframe = clean_infinite_and_null_values(dataframe)
    
    # Step 4: Parse and chronologically sort by timestamp
    dataframe = parse_and_sort_timestamps(dataframe)
    
    # Step 5: Standardize attack labels and generate binary indicator
    dataframe = standardize_labels(dataframe)
    
    print(f"  [Complete] Clean rows: {len(dataframe):,}")
    print(f"  [Time Range] From {dataframe['Timestamp'].min()} to {dataframe['Timestamp'].max()}")
    print(f"  [Attack Breakdown] {dict(dataframe['Attack_Category'].value_counts())}")
    
    return dataframe
