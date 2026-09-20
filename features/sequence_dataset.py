import os
import glob
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset
from features.windowing import STATE_FEATURE_NAMES, STATE_VECTOR_DIM


class NetworkStateScaler:
    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "NetworkStateScaler":
        X_arr = np.asarray(X, dtype=np.float32)
        self.mean_ = np.mean(X_arr, axis=0)
        std = np.std(X_arr, axis=0)
        self.scale_ = np.where(std < 1e-7, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float32)
        return (X_arr - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float32)
        return (X_arr * self.scale_) + self.mean_

    def save(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "NetworkStateScaler":
        with open(filepath, "rb") as f:
            scaler = pickle.load(f)
        return scaler


class NetworkSequenceDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y_state: np.ndarray,
        y_binary: np.ndarray,
        y_class: np.ndarray,
        timestamps: Optional[List[Dict]] = None
    ):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y_state = torch.tensor(y_state, dtype=torch.float32)
        self.y_binary = torch.tensor(y_binary, dtype=torch.float32)
        self.y_class = torch.tensor(y_class, dtype=torch.long)
        self.timestamps = timestamps or []
        
    def __len__(self) -> int:
        return len(self.x)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y_state[idx], self.y_binary[idx], self.y_class[idx]


def split_session_chronologically(
    session_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_windows = len(session_df)
    
    n_train = int(total_windows * train_ratio)
    n_val = int(total_windows * val_ratio)
    
    train_df = session_df.iloc[:n_train].copy().reset_index(drop=True)
    val_df = session_df.iloc[n_train:n_train + n_val].copy().reset_index(drop=True)
    test_df = session_df.iloc[n_train + n_val:].copy().reset_index(drop=True)
    
    return train_df, val_df, test_df


def build_sliding_sequences_for_partition(
    partition_df: pd.DataFrame,
    history_len_m: int = 10,
    forecast_horizon_k: int = 5,
    session_name: str = "unknown"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    total_windows = len(partition_df)
    required_windows = history_len_m + forecast_horizon_k
    
    if total_windows < required_windows:
        empty_x = np.empty((0, history_len_m, STATE_VECTOR_DIM), dtype=np.float32)
        empty_y_state = np.empty((0, STATE_VECTOR_DIM), dtype=np.float32)
        empty_y_binary = np.empty((0, forecast_horizon_k), dtype=np.float32)
        empty_y_class = np.empty((0, forecast_horizon_k), dtype=np.int64)
        return empty_x, empty_y_state, empty_y_binary, empty_y_class, []
        
    feature_matrix = partition_df[STATE_FEATURE_NAMES].values.astype(np.float32)
    is_attack_array = partition_df["is_attack"].values.astype(np.float32)
    class_index_array = partition_df["class_index"].values.astype(np.int64)
    
    start_times = partition_df["window_start"].tolist()
    end_times = partition_df["window_end"].tolist()
    
    x_list = []
    y_state_list = []
    y_binary_list = []
    y_class_list = []
    timestamp_meta_list = []
    
    num_sequences = total_windows - history_len_m - forecast_horizon_k + 1
    
    for i in range(num_sequences):
        x_seq = feature_matrix[i:i + history_len_m]
        y_state = feature_matrix[i + history_len_m]
        y_binary = is_attack_array[i + history_len_m:i + history_len_m + forecast_horizon_k]
        y_class = class_index_array[i + history_len_m:i + history_len_m + forecast_horizon_k]
        
        x_list.append(x_seq)
        y_state_list.append(y_state)
        y_binary_list.append(y_binary)
        y_class_list.append(y_class)
        
        timestamp_meta_list.append({
            "session": session_name,
            "history_start": str(start_times[i]),
            "history_end": str(end_times[i + history_len_m - 1]),
            "target_start": str(start_times[i + history_len_m]),
            "target_end": str(end_times[i + history_len_m + forecast_horizon_k - 1]),
        })
        
    x_arr = np.array(x_list, dtype=np.float32)
    y_state_arr = np.array(y_state_list, dtype=np.float32)
    y_binary_arr = np.array(y_binary_list, dtype=np.float32)
    y_class_arr = np.array(y_class_list, dtype=np.int64)
    
    return x_arr, y_state_arr, y_binary_arr, y_class_arr, timestamp_meta_list


def build_full_sequence_dataset(
    state_vectors_directory: str = "Datasets/state_vectors",
    output_directory: str = "Datasets/sequences",
    history_len_m: int = 10,
    forecast_horizon_k: int = 5,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Dict:
    os.makedirs(output_directory, exist_ok=True)
    
    parquet_files = sorted(glob.glob(os.path.join(state_vectors_directory, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No state vector Parquet files found in: {state_vectors_directory}")
        
    print("=" * 75)
    print("STAGE 4: Sequence Dataset Construction & Chronological Splitting")
    print(f"History (m) = {history_len_m} windows | Horizon (K) = {forecast_horizon_k} windows")
    print(f"Split Ratios: Train={train_ratio:.0%}, Val={val_ratio:.0%}, Test={test_ratio:.0%}")
    print("=" * 75)
    
    raw_train_dfs = []
    raw_val_dfs = []
    raw_test_dfs = []
    session_names = []
    
    for file_path in parquet_files:
        session_name = os.path.basename(file_path).split(".parquet")[0]
        session_names.append(session_name)
        df = pd.read_parquet(file_path)
        
        train_df, val_df, test_df = split_session_chronologically(
            session_df=df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
        
        raw_train_dfs.append(train_df)
        raw_val_dfs.append(val_df)
        raw_test_dfs.append(test_df)
        
        print(f"  [Split] {session_name[:40]}: Total={len(df)}, Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
    print("\n[Normalization] Fitting StandardScaler on combined Training state vectors...")
    combined_train_features = pd.concat([df[STATE_FEATURE_NAMES] for df in raw_train_dfs], ignore_index=True)
    
    scaler = NetworkStateScaler()
    scaler.fit(combined_train_features.values)
    
    scaler_path = os.path.join(output_directory, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  [Saved] Scaler saved to: {scaler_path}")
    
    def normalize_split(df_list: List[pd.DataFrame]) -> List[pd.DataFrame]:
        normalized_dfs = []
        for df in df_list:
            df_copy = df.copy()
            feature_vals = df_copy[STATE_FEATURE_NAMES].values
            normalized_vals = scaler.transform(feature_vals)
            df_copy[STATE_FEATURE_NAMES] = normalized_vals
            normalized_dfs.append(df_copy)
        return normalized_dfs
        
    norm_train_dfs = normalize_split(raw_train_dfs)
    norm_val_dfs = normalize_split(raw_val_dfs)
    norm_test_dfs = normalize_split(raw_test_dfs)
    
    def build_pool_sequences(df_list: List[pd.DataFrame], split_name: str):
        all_x, all_y_state, all_y_binary, all_y_class = [], [], [], []
        all_timestamps = []
        
        for df, s_name in zip(df_list, session_names):
            x, y_s, y_b, y_c, ts = build_sliding_sequences_for_partition(
                partition_df=df,
                history_len_m=history_len_m,
                forecast_horizon_k=forecast_horizon_k,
                session_name=s_name
            )
            if len(x) > 0:
                all_x.append(x)
                all_y_state.append(y_s)
                all_y_binary.append(y_b)
                all_y_class.append(y_c)
                all_timestamps.extend(ts)
                
        concat_x = np.concatenate(all_x, axis=0) if all_x else np.empty((0, history_len_m, STATE_VECTOR_DIM))
        concat_y_state = np.concatenate(all_y_state, axis=0) if all_y_state else np.empty((0, STATE_VECTOR_DIM))
        concat_y_binary = np.concatenate(all_y_binary, axis=0) if all_y_binary else np.empty((0, forecast_horizon_k))
        concat_y_class = np.concatenate(all_y_class, axis=0) if all_y_class else np.empty((0, forecast_horizon_k))
        
        print(f"  [{split_name} Pool] Built {len(concat_x):,} sequences. Input shape: {concat_x.shape}")
        
        archive_path = os.path.join(output_directory, f"{split_name.lower()}_sequences.npz")
        np.savez_compressed(
            archive_path,
            x=concat_x,
            y_state=concat_y_state,
            y_binary=concat_y_binary,
            y_class=concat_y_class
        )
        print(f"  [Saved] Exported {split_name} pool to: {archive_path}")
        
        return concat_x, concat_y_state, concat_y_binary, concat_y_class, all_timestamps

    print("\n[Sequencing] Building sliding-window sequences...")
    train_x, train_y_s, train_y_b, train_y_c, train_ts = build_pool_sequences(norm_train_dfs, "Train")
    val_x, val_y_s, val_y_b, val_y_c, val_ts = build_pool_sequences(norm_val_dfs, "Validation")
    test_x, test_y_s, test_y_b, test_y_c, test_ts = build_pool_sequences(norm_test_dfs, "Test")
    
    metadata = {
        "history_len_m": history_len_m,
        "forecast_horizon_k": forecast_horizon_k,
        "state_dim": STATE_VECTOR_DIM,
        "state_features": STATE_FEATURE_NAMES,
        "train_samples": len(train_x),
        "val_samples": len(val_x),
        "test_samples": len(test_x),
        "total_samples": len(train_x) + len(val_x) + len(test_x),
        "train_attack_rate": float((train_y_b[:, 0] == 1).mean()) if len(train_x) > 0 else 0.0,
        "val_attack_rate": float((val_y_b[:, 0] == 1).mean()) if len(val_x) > 0 else 0.0,
        "test_attack_rate": float((test_y_b[:, 0] == 1).mean()) if len(test_x) > 0 else 0.0,
    }
    
    metadata_path = os.path.join(output_directory, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"\n[Metadata] Exported metadata to: {metadata_path}")
    
    return {
        "train": (train_x, train_y_s, train_y_b, train_y_c, train_ts),
        "val": (val_x, val_y_s, val_y_b, val_y_c, val_ts),
        "test": (test_x, test_y_s, test_y_b, test_y_c, test_ts),
        "scaler": scaler,
        "metadata": metadata
    }
