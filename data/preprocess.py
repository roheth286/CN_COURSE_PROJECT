import os
import glob
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional

LABEL_MAPPING: Dict[str, str] = {
    "BENIGN": "BENIGN",
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "Heartbleed": "DoS",
    "DDoS": "DDoS",
    "PortScan": "Port Scan",
    "Web Attack - Brute Force": "Web Attack",
    "Web Attack \x96 Brute Force": "Web Attack",
    "Web Attack - XSS": "Web Attack",
    "Web Attack \x96 XSS": "Web Attack",
    "Web Attack - Sql Injection": "Web Attack",
    "Web Attack \x96 Sql Injection": "Web Attack",
    "Bot": "Botnet",
    "Infiltration": "Infiltration",
}

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
    cleaned_column_names = []
    for column_name in dataframe.columns:
        cleaned_column_names.append(column_name.strip())
    dataframe.columns = cleaned_column_names
    return dataframe


def drop_corrupted_and_empty_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    initial_row_count = len(dataframe)
    dataframe = dataframe.dropna(how="all")
    
    if "Timestamp" in dataframe.columns:
        dataframe = dataframe.dropna(subset=["Timestamp"])
        
    if "Label" in dataframe.columns:
        dataframe = dataframe.dropna(subset=["Label"])
        dataframe = dataframe[dataframe["Label"].astype(str).str.strip() != ""]
        
    final_row_count = len(dataframe)
    dropped_count = initial_row_count - final_row_count
    if dropped_count > 0:
        print(f"  [Clean] Dropped {dropped_count:,} empty/corrupted rows.")
        
    return dataframe


def clean_infinite_and_null_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    
    for column_name in numeric_columns:
        column_series = dataframe[column_name]
        has_infinite = np.isinf(column_series).any()
        if has_infinite:
            dataframe[column_name] = dataframe[column_name].replace([np.inf, -np.inf], np.nan)
            valid_numbers = dataframe[column_name].dropna()
            if len(valid_numbers) > 0:
                cap_value = valid_numbers.quantile(0.999)
                dataframe[column_name] = dataframe[column_name].fillna(cap_value)
            else:
                dataframe[column_name] = dataframe[column_name].fillna(0.0)
                
        if dataframe[column_name].isnull().any():
            median_value = dataframe[column_name].median()
            dataframe[column_name] = dataframe[column_name].fillna(median_value)
            
    return dataframe


def parse_and_sort_timestamps(dataframe: pd.DataFrame) -> pd.DataFrame:
    if "Timestamp" not in dataframe.columns:
        raise ValueError("The dataframe does not contain a 'Timestamp' column.")
        
    dataframe["Timestamp"] = pd.to_datetime(
        dataframe["Timestamp"].astype(str).str.strip(),
        format="mixed",
        dayfirst=True,
        errors="coerce"
    )
    
    dataframe = dataframe.dropna(subset=["Timestamp"])
    dataframe = dataframe.sort_values(by="Timestamp", ascending=True)
    dataframe = dataframe.reset_index(drop=True)
    return dataframe


def standardize_labels(dataframe: pd.DataFrame) -> pd.DataFrame:
    if "Label" not in dataframe.columns:
        raise ValueError("The dataframe does not contain a 'Label' column.")
        
    raw_labels = dataframe["Label"].astype(str).str.strip()
    standardized_categories = []
    is_attack_flags = []
    class_indices = []
    
    for raw_label in raw_labels:
        matched_category = LABEL_MAPPING.get(raw_label, None)
        if matched_category is None:
            if "web attack" in raw_label.lower():
                matched_category = "Web Attack"
            elif "benign" in raw_label.lower():
                matched_category = "BENIGN"
            else:
                matched_category = "BENIGN"
                
        standardized_categories.append(matched_category)
        
        if matched_category == "BENIGN":
            is_attack_flags.append(0)
        else:
            is_attack_flags.append(1)
            
        class_indices.append(CLASS_TO_INDEX.get(matched_category, 0))
        
    dataframe["Attack_Category"] = standardized_categories
    dataframe["Is_Attack"] = is_attack_flags
    dataframe["Class_Index"] = class_indices
    return dataframe


def load_and_preprocess_single_file(file_path: str) -> pd.DataFrame:
    file_name = os.path.basename(file_path)
    print(f"\n[Processing] Loading file: {file_name}")
    
    dataframe = pd.read_csv(
        file_path,
        encoding="latin-1",
        low_memory=False
    )
    print(f"  [Loaded] Raw rows: {len(dataframe):,} | Columns: {len(dataframe.columns)}")
    
    dataframe = sanitize_column_names(dataframe)
    dataframe = drop_corrupted_and_empty_rows(dataframe)
    dataframe = clean_infinite_and_null_values(dataframe)
    dataframe = parse_and_sort_timestamps(dataframe)
    dataframe = standardize_labels(dataframe)
    
    print(f"  [Complete] Clean rows: {len(dataframe):,}")
    print(f"  [Time Range] From {dataframe['Timestamp'].min()} to {dataframe['Timestamp'].max()}")
    print(f"  [Attack Breakdown] {dict(dataframe['Attack_Category'].value_counts())}")
    
    return dataframe
